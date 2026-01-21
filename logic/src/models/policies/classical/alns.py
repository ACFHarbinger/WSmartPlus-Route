"""
ALNS Policy wrapper for RL4CO.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.models.policies.classical.adaptive_large_neighborhood_search import ALNSSolver
from logic.src.models.policies.classical.alns_aux.params import ALNSParams


class ALNSPolicy(ConstructivePolicy):
    """
    ALNS-based Policy wrapper.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        max_iterations: int = 1000,
        **kwargs,
    ):
        super().__init__(env_name=env_name, **kwargs)
        self.params = ALNSParams(
            time_limit=time_limit,
            max_iterations=max_iterations,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "greedy",  # Ignored for ALNS
        **kwargs,
    ) -> dict:
        """
        Solve instances in the batch using ALNS.
        Note: Currently solves sequentially if batch size > 1.
        """
        batch_size = td.batch_size[0]
        device = td.device

        all_actions = []
        all_rewards = []

        # We need dist_matrix, demands, capacity, R, C
        # These are usually in the env or td

        # VRPP specific extraction
        # Reward is usually prize - cost
        R = getattr(env, "prize_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        for i in range(batch_size):
            td_idx = td[i]

            # Extract data
            dist_matrix = td_idx["locs"]  # Simplified, usually needs distance calc or pre-computed
            # If locs are coords, we need to compute distance matrix
            if dist_matrix.dim() == 2 and dist_matrix.shape[-1] == 2:
                from logic.src.utils.functions.function import compute_distance_matrix

                dist_matrix_np = compute_distance_matrix(dist_matrix.cpu().numpy(), method="og")
            else:
                dist_matrix_np = dist_matrix.cpu().numpy()

            demands = td_idx.get("demand", td_idx.get("prize", torch.zeros(dist_matrix_np.shape[0]))).cpu().numpy()
            demands_dict = {j: float(demands[j]) for j in range(len(demands))}
            capacity = float(td_idx.get("capacity", torch.tensor(1.0)).cpu())

            # Optional initial solution refinement
            init_routes = None
            if kwargs.get("initial_solution") is not None:
                # Convert tensor [batch, seq_len] to List[List[int]] for this sample
                init_actions = kwargs["initial_solution"][i]
                # Split by 0 (depot) and remove zeros
                init_routes = []
                curr_route = []
                for action in init_actions.tolist():
                    if action == 0:
                        if curr_route:
                            init_routes.append(curr_route)
                        curr_route = []
                    else:
                        curr_route.append(int(action))
                if curr_route:
                    init_routes.append(curr_route)

            solver = ALNSSolver(
                dist_matrix=dist_matrix_np, demands=demands_dict, capacity=capacity, R=R, C=C, params=self.params
            )

            routes, profit, cost = solver.solve(initial_solution=init_routes)

            # Convert routes (list of lists) to actions (tensor)
            # RL4CO usually expects [batch, seq_len] actions
            # We flatten routes and add depot (0) at start/between/end
            flat_actions = []
            for route in routes:
                flat_actions.append(0)
                flat_actions.extend(route)
            flat_actions.append(0)

            all_actions.append(torch.tensor(flat_actions, device=device))
            all_rewards.append(torch.tensor(profit, device=device))

        # Pad actions to same length if needed
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        return {
            "reward": torch.stack(all_rewards),
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
        }
