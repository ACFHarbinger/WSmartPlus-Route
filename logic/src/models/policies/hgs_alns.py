"""
HGS-ALNS Hybrid Policy for RL4CO.
"""

from __future__ import annotations

from typing import Any

import torch
from joblib import Parallel, delayed, parallel_backend
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.hgs import VectorizedHGS
from logic.src.policies.hgs_alns import HGSALNSSolver
from logic.src.policies.hybrid_genetic_search.params import HGSParams


class VectorizedHGSALNS(VectorizedHGS):
    """
    HGS-based Policy wrapper that uses ALNS for education phase.

    This implementation combines the logic of HGSPolicy for RL4CO integration
    with ALNS-based education.
    """

    def __init__(
        self,
        env_name: str | None = "vrpp",
        time_limit: float = 0.1,
        population_size: int = 20,
        n_generations: int = 15,
        elite_size: int = 2,
        max_vehicles: int = 0,
        alns_education_iterations: int = 5,
        **kwargs: Any,
    ):
        """Initialize VectorizedHGSALNS."""
        super().__init__(
            env_name=env_name,
            time_limit=time_limit,
            population_size=population_size,
            n_generations=n_generations,
            elite_size=elite_size,
            max_vehicles=max_vehicles,
            **kwargs,
        )
        self.alns_education_iterations = alns_education_iterations

    def solve(self, dist_matrix, demands, capacity, **kwargs):
        """
        Solve a single instance using the scalar HGSALNSSolver.

        This overrides the standard solve logic to use the hybrid solver.
        """
        if isinstance(dist_matrix, torch.Tensor):
            dist_matrix = dist_matrix.cpu().numpy()
        if isinstance(demands, torch.Tensor):
            # If batch dim is present, take first instance
            if demands.dim() > 1:
                demands = demands[0]
            demands_dict = {i: demands[i].item() for i in range(len(demands))}
        else:
            demands_dict = demands

        params = HGSParams(
            time_limit=float(self.time_limit),
            population_size=self.population_size,
            elite_size=self.elite_size,
            max_vehicles=self.max_vehicles,
        )

        solver = HGSALNSSolver(
            dist_matrix=dist_matrix,
            demands=demands_dict,
            capacity=float(capacity),
            R=kwargs.get("R", 1.0),
            C=kwargs.get("C", 1.0),
            params=params,
            alns_education_iterations=self.alns_education_iterations,
        )

        return solver.solve()

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        strategy: str = "greedy",
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Robust forward pass for imitation learning.

        Iterates over the batch and uses the scalar HGS-ALNS solver which includes
        a robust fallback mechanism in LinearSplit to ensure finite rewards.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # Extract data from TensorDict
        customers = td["locs"]  # (batch, num_nodes, 2)
        depot = td["depot"].unsqueeze(1)  # (batch, 1, 2)
        locs = torch.cat([depot, customers], dim=1)  # (batch, num_nodes + 1, 2)

        # Extract waste (demands)
        waste_at_nodes = td.get("waste", torch.zeros(batch_size, locs.shape[1] - 1, device=device))
        waste = torch.cat([torch.zeros(batch_size, 1, device=device), waste_at_nodes], dim=1)
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * 10.0)

        # Compute Euclidean distance matrix if locations provided
        if locs.dim() == 3 and locs.shape[-1] == 2:
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        R = getattr(env, "waste_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        # Solve the batch in parallel
        # Note: We must use 'multiprocessing' backend as requested, but limit concurrency
        # to prevent UI freeze (Rich needs cycles) and system starvation.
        # Spawning too many processes (n_jobs=-1) inside DataLoader workers causes deadlock/freeze.
        n_jobs = min(8, batch_size)
        with parallel_backend("multiprocessing"):
            results: list[tuple[list[list[int]], float, float]] = Parallel(n_jobs=n_jobs)(
                delayed(self.solve)(
                    dist_matrix=dist_matrix[b],
                    demands=waste[b],
                    capacity=capacity[b].item() if isinstance(capacity, torch.Tensor) else capacity,
                    R=R,
                    C=C,
                )
                for b in range(batch_size)
            )

        all_actions = []
        all_rewards = []
        all_costs = []

        for routes, profit_score, cost in results:
            # Convert routes to action sequence
            flat_actions = []
            for route in routes:
                flat_actions.append(0)
                flat_actions.extend(route)
            flat_actions.append(0)

            all_actions.append(torch.tensor(flat_actions, device=device, dtype=torch.long))
            all_costs.append(torch.tensor(cost, device=device))
            all_rewards.append(torch.tensor(profit_score, device=device))

        # Pad actions
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        return {
            "reward": torch.stack(all_rewards),
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
            "cost": torch.stack(all_costs),
        }
