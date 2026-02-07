"""
ACO Decoder: Ant colony solution construction from heatmaps.

Uses pheromone-guided ant colony optimization to construct solutions.
"""

from __future__ import annotations

from typing import Dict

import torch
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.common.nonautoregressive import NonAutoregressiveDecoder


class ACODecoder(NonAutoregressiveDecoder):
    """
    Ant Colony Optimization decoder.

    Constructs solutions using heatmap-guided pheromones.
    """

    def __init__(
        self,
        n_ants: int = 20,
        n_iterations: int = 1,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        use_local_search: bool = True,
        **kwargs,
    ):
        """
        Initialize ACODecoder.

        Args:
            n_ants: Number of ants per construction.
            n_iterations: Number of ACO iterations.
            alpha: Pheromone importance weight.
            beta: Heuristic (distance) importance weight.
            rho: Pheromone evaporation rate.
            use_local_search: Whether to apply 2-opt local search.
        """
        super().__init__(**kwargs)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.use_local_search = use_local_search

    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct solutions using ACO with heatmap pheromones.

        Args:
            td: TensorDict with problem instance.
            heatmap: Edge heatmap [batch, n, n] (log probabilities).
            env: Environment for reward calculation.
            num_starts: Number of independent ACO runs.

        Returns:
            Dictionary with actions, log_likelihood, reward.
        """
        batch_size, num_nodes, _ = heatmap.shape

        # Convert log heatmap to pheromone
        pheromone = heatmap.exp()  # [batch, n, n]

        # Compute distance matrix for heuristic
        locs = td["locs"]  # [batch, n, 2]
        dist_matrix = torch.cdist(locs, locs, p=2)  # [batch, n, n]
        # Heuristic: inverse distance (avoid div by zero)
        heuristic = 1.0 / (dist_matrix + 1e-10)

        # Combined probability matrix
        prob_matrix = (pheromone**self.alpha) * (heuristic**self.beta)

        # Run ants
        best_tours, best_costs, log_probs = self._run_ants(prob_matrix, dist_matrix, td, env)

        # Optional local search
        if self.use_local_search:
            best_tours, best_costs = self._two_opt(best_tours, dist_matrix)

        return {
            "actions": best_tours,
            "reward": -best_costs,  # Reward is negative cost
            "log_likelihood": log_probs,
        }

    def _run_ants(
        self,
        prob_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
        td: TensorDict,
        env: RL4COEnvBase,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run ant colony to construct solutions.

        Args:
            prob_matrix: Combined pheromone * heuristic [batch, n, n].
            dist_matrix: Distance matrix [batch, n, n].
            td: TensorDict with problem instance.
            env: Environment.

        Returns:
            Tuple of (best_tours, best_costs, log_probs).
        """
        batch_size, num_nodes, _ = prob_matrix.shape
        device = prob_matrix.device

        best_tours = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        best_costs = torch.full((batch_size,), float("inf"), device=device)
        total_log_probs = torch.zeros(batch_size, device=device)

        for _ in range(self.n_ants):
            # Construct tour for each ant
            tours, costs, log_probs = self._construct_tour(prob_matrix, dist_matrix)

            # Update best
            improved = costs < best_costs
            best_costs = torch.where(improved, costs, best_costs)
            best_tours = torch.where(improved.unsqueeze(-1), tours, best_tours)
            total_log_probs = total_log_probs + log_probs

        return best_tours, best_costs, total_log_probs / self.n_ants

    def _construct_tour(
        self,
        prob_matrix: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct a single tour using probabilistic selection.

        Args:
            prob_matrix: Selection probabilities [batch, n, n].
            dist_matrix: Distance matrix [batch, n, n].

        Returns:
            Tuple of (tour, cost, log_prob).
        """
        batch_size, num_nodes, _ = prob_matrix.shape
        device = prob_matrix.device

        # Start from depot (node 0)
        tour = [torch.zeros(batch_size, dtype=torch.long, device=device)]
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        visited[:, 0] = True

        total_log_prob = torch.zeros(batch_size, device=device)
        current = tour[0]

        for _ in range(num_nodes - 1):
            # Get probabilities for next node
            probs = prob_matrix[torch.arange(batch_size, device=device), current]
            probs = probs.masked_fill(visited, 0.0)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

            # Sample next node
            next_node = torch.multinomial(probs, 1).squeeze(-1)

            # Record
            tour.append(next_node)
            visited[torch.arange(batch_size, device=device), next_node] = True
            total_log_prob = total_log_prob + torch.log(
                probs[torch.arange(batch_size, device=device), next_node] + 1e-10
            )
            current = next_node

        # Stack tour
        tour_tensor = torch.stack(tour, dim=1)  # [batch, n]

        # Compute tour cost
        cost = self._compute_tour_cost(tour_tensor, dist_matrix)

        return tour_tensor, cost, total_log_prob

    def _compute_tour_cost(
        self,
        tour: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total tour distance."""
        batch_size, num_nodes = tour.shape
        device = tour.device

        # Gather distances for each edge in tour
        from_nodes = tour
        to_nodes = torch.roll(tour, -1, dims=1)

        batch_idx = torch.arange(batch_size, device=device).unsqueeze(-1)
        costs = dist_matrix[batch_idx, from_nodes, to_nodes].sum(dim=-1)
        return costs

    def _two_opt(
        self,
        tours: torch.Tensor,
        dist_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2-opt local search to improve tours.

        Args:
            tours: Tour sequences [batch, n].
            dist_matrix: Distance matrix [batch, n, n].

        Returns:
            Improved (tours, costs).
        """
        # Simple 2-opt: try all swaps, keep improvements
        batch_size, num_nodes = tours.shape

        improved = True
        best_tours = tours.clone()
        best_costs = self._compute_tour_cost(best_tours, dist_matrix)

        max_iters = 10  # Limit iterations for efficiency
        for _ in range(max_iters):
            improved = False
            for i in range(1, num_nodes - 1):
                for j in range(i + 1, num_nodes):
                    # Reverse segment [i, j]
                    new_tours = best_tours.clone()
                    new_tours[:, i : j + 1] = torch.flip(best_tours[:, i : j + 1], dims=[1])
                    new_costs = self._compute_tour_cost(new_tours, dist_matrix)

                    # Keep if improved
                    improved_mask = new_costs < best_costs
                    if improved_mask.any():
                        improved = True
                        best_costs = torch.where(improved_mask, new_costs, best_costs)
                        best_tours = torch.where(improved_mask.unsqueeze(-1), new_tours, best_tours)

            if not improved:
                break

        return best_tours, best_costs
