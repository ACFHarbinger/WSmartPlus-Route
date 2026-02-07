"""
Vectorized HGS Algorithm.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import torch
from logic.src.models.policies.classical.hgs.crossover import vectorized_ordered_crossover
from logic.src.models.policies.classical.hgs.population import VectorizedPopulation
from logic.src.models.policies.classical.local_search import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
from logic.src.models.policies.classical.shared import vectorized_linear_split


class VectorizedHGS:
    """
    Main class for the Vectorized Hybrid Genetic Search (HGS) algorithm.
    Orchestrates the evolution process, including initialization, selection, crossover,
    local search (education), and survivor selection.

    Args:
        dist_matrix (torch.Tensor): Distance matrix.
        demands (torch.Tensor): Demands tensor.
        vehicle_capacity (float/torch.Tensor): Vehicle capacity.
        time_limit (float): Time limit for the search in seconds.
        device: Torch device.
    """

    def __init__(
        self,
        dist_matrix: torch.Tensor,
        demands: torch.Tensor,
        vehicle_capacity: Any,
        time_limit: float = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize the HGS solver.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = torch.device(device)

    def solve(
        self,
        initial_solutions: torch.Tensor,
        n_generations: int = 50,
        population_size: int = 10,
        elite_size: int = 5,
        time_limit: Optional[float] = None,
        max_vehicles: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the HGS algorithm starting from a set of initial solutions (Expert Imitation Mode).

        Args:
            initial_solutions (torch.Tensor): Initial solutions (giant tours) (B, N).
            n_generations (int): Number of generations to run.
            population_size (int): Size of the genetic population.
            elite_size (int): Number of elite individuals to preserve during restarting.
            time_limit (float, optional): Override time limit in seconds.
            max_vehicles (int, optional): Maximum number of vehicles allowed (0 for unlimited).

        Returns:
            tuple: (best_routes, best_cost)
        """
        B, N = initial_solutions.size()
        start_time = time.time()

        # Initial Evaluation
        _, costs = vectorized_linear_split(
            initial_solutions,
            self.dist_matrix,
            self.demands,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )

        pop = VectorizedPopulation(population_size, self.device)
        pop.initialize(initial_solutions, costs)

        no_improv = 0
        best_cost_tracker = pop.costs.min().item()

        for gen in range(n_generations):
            if time_limit is not None and (time.time() - start_time > time_limit):
                break

            # Selection
            p1, p2 = pop.get_parents(n_offspring=1)
            p1 = p1.squeeze(1)
            p2 = p2.squeeze(1)

            # Crossover
            offspring_giant = vectorized_ordered_crossover(p1, p2)

            # Evaluation
            routes_list, split_costs = vectorized_linear_split(
                offspring_giant,
                self.dist_matrix,
                self.demands,
                self.vehicle_capacity,
                max_vehicles=max_vehicles,
            )

            # Education (Local Search)
            max_l = max(len(r) for r in routes_list)
            offspring_routes = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
            for b in range(B):
                r = routes_list[b]
                offspring_routes[b, : len(r)] = torch.tensor(r, device=self.device)

            if max_l > 2:
                improved_routes = vectorized_two_opt(offspring_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_three_opt(improved_routes, self.dist_matrix, max_iterations=20)
                improved_routes = vectorized_swap(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_relocate(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_two_opt_star(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_swap_star(improved_routes, self.dist_matrix, max_iterations=50)

                # Recalculate cost
                from_n = improved_routes[:, :-1]
                to_n = improved_routes[:, 1:]

                # Flexible distance matrix support (batch, non-batch, 2D, 3D)
                if self.dist_matrix.dim() == 3 and self.dist_matrix.size(0) == B:
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = self.dist_matrix[batch_ids, from_n, to_n]
                else:
                    expanded_dm = self.dist_matrix
                    if expanded_dm.dim() == 2:
                        expanded_dm = expanded_dm.unsqueeze(0)
                    if expanded_dm.size(0) == 1 and B > 1:
                        expanded_dm = expanded_dm.expand(B, -1, -1)
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = expanded_dm[batch_ids, from_n, to_n]

                improved_costs = dists.sum(dim=1)
            else:
                improved_routes = offspring_routes
                improved_costs = split_costs

            # Reconstruct Giant Tour
            giant_candidates = torch.zeros((B, N), dtype=torch.long, device=self.device)
            improved_routes_cpu = improved_routes.cpu().numpy()

            for b in range(B):
                r_full = improved_routes_cpu[b]
                g_tour = r_full[r_full != 0]
                if len(g_tour) == N:
                    giant_candidates[b, :] = torch.tensor(g_tour, device=self.device)
                else:
                    giant_candidates[b, :] = offspring_giant[b, :]

            # Survivor Selection
            pop.add_individuals(giant_candidates, improved_costs)

            # Stagnation Check
            current_best = pop.costs.min().item()
            if current_best < best_cost_tracker - 1e-4:
                best_cost_tracker = current_best
                no_improv = 0
            else:
                no_improv += 1

            if no_improv > 50:
                k = elite_size
                # Keep Elite
                sorted_idx = torch.argsort(pop.biased_fitness, dim=1)[:, :k]
                elite_pop = torch.gather(pop.population, 1, sorted_idx.unsqueeze(2).expand(-1, -1, N))
                elite_cost = torch.gather(pop.costs, 1, sorted_idx)

                # New candidates (Random Permutations)
                n_rest = pop.max_size - k
                new_pop = torch.zeros((B, n_rest, N), dtype=torch.long, device=self.device)
                for b in range(B):
                    for i in range(n_rest):
                        new_pop[b, i] = torch.randperm(N, device=self.device) + 1

                # Eval new pop
                # Flatten (B, n_rest, N) -> (B*n_rest, N)
                b_sz, n_rst, n_nds = new_pop.shape
                flat_pop = new_pop.view(b_sz * n_rst, n_nds)

                flat_dist = self.dist_matrix.repeat_interleave(n_rst, dim=0)
                flat_demands = self.demands.repeat_interleave(n_rst, dim=0)
                flat_cap = self.vehicle_capacity
                if isinstance(flat_cap, torch.Tensor) and flat_cap.dim() > 0:
                    flat_cap = flat_cap.repeat_interleave(n_rst, dim=0)

                _, new_costs_flat = vectorized_linear_split(
                    flat_pop,
                    flat_dist,
                    flat_demands,
                    flat_cap,
                    max_vehicles=max_vehicles,
                )
                new_costs = new_costs_flat.view(b_sz, n_rst)

                # Update pop directly
                pop.population = torch.cat([elite_pop, new_pop], dim=1)
                pop.costs = torch.cat([elite_cost, new_costs], dim=1)
                pop.compute_biased_fitness()
                no_improv = 0

        best_cost_pop, best_idx = torch.min(pop.costs, dim=1)
        best_giant = pop.population[torch.arange(B), best_idx]
        best_routes, best_final_costs = vectorized_linear_split(
            best_giant,
            self.dist_matrix,
            self.demands,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )
        return best_routes, best_final_costs
