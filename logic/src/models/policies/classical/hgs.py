"""
HGS Policy wrapper for RL4CO.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy
from logic.src.models.policies.classical.hgs_aux.types import HGSParams
from logic.src.models.policies.classical.hybrid_genetic_search import HGSSolver


class HGSPolicy(ConstructivePolicy):
    """
    HGS-based Policy wrapper.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        population_size: int = 50,
        max_vehicles: int = 0,
        **kwargs,
    ):
        super().__init__(env_name=env_name, **kwargs)
        self.params = HGSParams(
            time_limit=time_limit,
            population_size=population_size,
            max_vehicles=max_vehicles,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "greedy",
        **kwargs,
    ) -> dict:
        """
        Solve instances in the batch using HGS.
        """
        batch_size = td.batch_size[0]
        device = td.device

        all_actions = []
        all_rewards = []

        R = getattr(env, "prize_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        for i in range(batch_size):
            td_idx = td[i]

            # Extract data
            locs = td_idx["locs"]
            if locs.dim() == 2 and locs.shape[-1] == 2:
                import pandas as pd

                coords_df = pd.DataFrame(locs.cpu().numpy(), columns=["Lat", "Lng"])
                coords_df["ID"] = range(len(coords_df))
                from logic.src.pipeline.simulations.network import compute_distance_matrix

                dist_matrix_np = compute_distance_matrix(coords_df, method="ogd")
            else:
                dist_matrix_np = locs.cpu().numpy()

            demands = td_idx.get("demand", td_idx.get("prize", torch.zeros(dist_matrix_np.shape[0]))).cpu().numpy()
            demands_dict = {j: float(demands[j]) for j in range(len(demands))}
            capacity = float(td_idx.get("capacity", torch.tensor(1.0)).cpu())

            solver = HGSSolver(
                dist_matrix=dist_matrix_np, demands=demands_dict, capacity=capacity, R=R, C=C, params=self.params
            )

            routes, profit, cost = solver.solve()

            flat_actions = []
            for route in routes:
                flat_actions.append(0)
                flat_actions.extend(route)
            flat_actions.append(0)

            all_actions.append(torch.tensor(flat_actions, device=device))
            all_rewards.append(torch.tensor(profit, device=device))

        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        return {
            "reward": torch.stack(all_rewards),
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
        }
