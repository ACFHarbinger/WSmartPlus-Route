"""
ALNS Policy wrapper for RL4CO using vectorized implementation.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.constants.simulation import VEHICLE_CAPACITY
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy

from .adaptive_large_neighborhood_search import VectorizedALNS as VectorizedALNSEngine


class VectorizedALNS(ConstructivePolicy):
    """
    ALNS-based Policy wrapper using vectorized GPU-accelerated implementation.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        max_iterations: int = 500,
        max_vehicles: int = 0,
        **kwargs,
    ):
        """Initialize ALNSPolicy."""
        super().__init__(env_name=env_name, **kwargs)
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.max_vehicles = max_vehicles

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "greedy",  # Ignored for ALNS
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Solve instances in the batch using vectorized ALNS.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # Extract data
        customers = td["locs"]  # (batch, num_nodes, 2)
        depot = td["depot"].unsqueeze(1)  # (batch, 1, 2)
        locs = torch.cat([depot, customers], dim=1)  # (batch, num_nodes + 1, 2)

        device = locs.device
        num_nodes = locs.shape[1]

        # Compute distance matrix if needed
        if locs.dim() == 3 and locs.shape[-1] == 2:
            # Compute Euclidean distance matrix
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        # Extract demands
        prizes = td.get("prize", torch.zeros(batch_size, num_nodes - 1, device=device))
        demands_at_nodes = td.get("demand", prizes)
        # Prepend 0 for depot
        demands = torch.cat([torch.zeros(batch_size, 1, device=device), demands_at_nodes], dim=1)

        # Extract capacity
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * VEHICLE_CAPACITY)
        if capacity.dim() == 0:
            capacity = capacity.expand(batch_size)

        # Create initial solutions
        if kwargs.get("initial_solution") is not None:
            initial_solutions = kwargs["initial_solution"]
            # Ensure it's a giant tour (permutation of 1..num_nodes-1)
            # If it's already a giant tour, great. If it has 0s, we might need to filter.
            # For simplicity, we assume if provided it's usable or we fall back.
            if initial_solutions.size(1) != num_nodes - 1:
                # Fallback to random
                initial_solutions = torch.stack(
                    [torch.randperm(num_nodes - 1, device=device) + 1 for _ in range(batch_size)]
                )
        else:
            initial_solutions = torch.stack(
                [torch.randperm(num_nodes - 1, device=device) + 1 for _ in range(batch_size)]
            )

        solver = VectorizedALNSEngine(
            dist_matrix=dist_matrix,
            demands=demands,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=device,
        )

        routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_iterations=self.max_iterations,
            time_limit=self.time_limit,
            max_vehicles=kwargs.get("max_vehicles", self.max_vehicles),
        )

        # Convert routes to actions (padded tensor)
        all_actions = []
        for b in range(batch_size):
            route_nodes = routes_list[b]
            # Ensure it starts and ends with 0 if not already
            if not isinstance(route_nodes, torch.Tensor):
                route_nodes = torch.tensor(route_nodes, device=device)

            # vectorized_linear_split usually returns routes with 0s.
            # If not, we add them.
            if route_nodes.size(0) > 0:
                actions = route_nodes
            else:
                actions = torch.tensor([0, 0], device=device)
            all_actions.append(actions)

        # Pad actions
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        # Compute rewards (profit - cost)
        R = getattr(env, "prize_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        # Approximate rewards for now (similar to HGS implementation)
        all_rewards = []
        for b in range(batch_size):
            collected_nodes = set(all_actions[b].tolist()) - {0}
            profit = sum(demands[b, node].item() * R for node in collected_nodes if node < num_nodes)
            cost = costs[b].item() * C
            all_rewards.append(torch.tensor(profit - cost, device=device))

        return {
            "reward": torch.stack(all_rewards),
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": costs,
        }
