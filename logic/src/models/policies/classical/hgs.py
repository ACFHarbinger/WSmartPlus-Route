"""
HGS Policy wrapper for RL4CO using vectorized implementation.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.base import ConstructivePolicy

from .hybrid_genetic_search import VectorizedHGS


class HGSPolicy(ConstructivePolicy):
    """
    HGS-based Policy wrapper using vectorized GPU-accelerated implementation.

    This policy uses the vectorized Hybrid Genetic Search algorithm which
    processes entire batches in parallel on GPU for significant speedup.
    """

    def __init__(
        self,
        env_name: str,
        time_limit: float = 5.0,
        population_size: int = 50,
        n_generations: int = 50,
        elite_size: int = 5,
        max_vehicles: int = 0,
        **kwargs,
    ):
        """Initialize HGSPolicy with vectorized solver."""
        super().__init__(env_name=env_name, **kwargs)
        self.time_limit = time_limit
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_size = elite_size
        self.max_vehicles = max_vehicles

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "greedy",
        **kwargs,
    ) -> dict:
        """
        Solve instances in the batch using vectorized HGS.

        Processes the entire batch in parallel on GPU for improved performance.
        """
        batch_size = td.batch_size[0]
        device = td.device

        # Extract data
        locs = td["locs"]  # (batch, num_nodes, 2)
        num_nodes = locs.shape[1]

        # Compute distance matrix if needed
        if locs.dim() == 3 and locs.shape[-1] == 2:
            # Compute Euclidean distance matrix
            # (batch, num_nodes, 1, 2) - (batch, 1, num_nodes, 2) -> (batch, num_nodes, num_nodes)
            diff = locs.unsqueeze(2) - locs.unsqueeze(1)
            dist_matrix = torch.sqrt((diff**2).sum(dim=-1))
        else:
            dist_matrix = locs

        # Extract demands
        demands = td.get("demand", td.get("prize", torch.zeros(batch_size, num_nodes, device=device)))

        # Extract capacity
        capacity = td.get("capacity", torch.ones(batch_size, device=device) * 100.0)
        if capacity.dim() == 0:
            capacity = capacity.unsqueeze(0).expand(batch_size)

        # Create initial solutions (random permutations of non-depot nodes)
        # Nodes are indexed 1..num_nodes (0 is depot)
        initial_solutions = torch.zeros(batch_size, num_nodes - 1, dtype=torch.long, device=device)
        for b in range(batch_size):
            initial_solutions[b] = torch.randperm(num_nodes - 1, device=device) + 1

        # Initialize vectorized HGS solver
        solver = VectorizedHGS(
            dist_matrix=dist_matrix,
            demands=demands,
            vehicle_capacity=capacity,
            time_limit=self.time_limit,
            device=device,
        )

        # Run HGS
        routes_list, costs = solver.solve(
            initial_solutions=initial_solutions,
            n_generations=self.n_generations,
            population_size=self.population_size,
            elite_size=self.elite_size,
            time_limit=self.time_limit,
            max_vehicles=self.max_vehicles,
        )

        # Convert routes to actions format
        # routes_list is a list of routes (one per batch element)
        all_actions = []
        for b in range(batch_size):
            routes = routes_list[b] if isinstance(routes_list, list) else [routes_list[b].tolist()]

            # Flatten routes with depot separators
            flat_actions = []

            # Check if routes is a list of lists (multiple tours) or a flat list (single tour)
            if routes and isinstance(routes[0], int):
                # Single tour case: wrap in list
                routes = [routes]

            for route in routes:
                flat_actions.append(0)  # Start at depot
                if isinstance(route, (list, tuple)):
                    flat_actions.extend(route)
                elif isinstance(route, torch.Tensor):
                    flat_actions.extend(route.tolist())
                else:
                    # Fallback or error if route is not iterable (e.g. int)
                    # Ideally this shouldn't happen if we handled the single tour case
                    try:
                        flat_actions.extend(route)
                    except TypeError:
                        # If route is a single node (int), treat as length-1 route
                        flat_actions.append(route)
            flat_actions.append(0)  # Return to depot

            all_actions.append(torch.tensor(flat_actions, device=device, dtype=torch.long))

        # Compute rewards (profit - cost)
        R = getattr(env, "prize_weight", 1.0)
        C = getattr(env, "cost_weight", 1.0)

        all_rewards = []
        for b in range(batch_size):
            # Calculate profit from collected nodes
            collected_nodes = set(all_actions[b].tolist()) - {0}
            profit = sum(demands[b, node].item() * R for node in collected_nodes if node < num_nodes)

            # Cost is already computed
            cost = costs[b].item() * C

            reward = profit - cost
            all_rewards.append(torch.tensor(reward, device=device))

        # Pad actions to same length
        max_len = max(len(a) for a in all_actions)
        padded_actions = torch.stack(
            [torch.cat([a, torch.zeros(max_len - len(a), device=device, dtype=torch.long)]) for a in all_actions]
        )

        return {
            "reward": torch.stack(all_rewards),
            "actions": padded_actions,
            "log_likelihood": torch.zeros(batch_size, device=device),
        }
