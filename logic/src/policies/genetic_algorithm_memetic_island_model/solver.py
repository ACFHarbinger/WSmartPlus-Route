"""
Memetic Island Model Genetic Algorithm for VRPP.

Multi-population evolutionary algorithm with periodic migration between
sub-populations. Replaces metaphor-based sports algorithms (HVPL, LCA, SLC):
- "Teams" → Islands (independent sub-populations)
- "Matches/Competition" → Fitness evaluations and tournament selection
- "Seasons" → Generations
- "Coaching" → Local search improvement operator
- "Relegation/Promotion" → Population replacement via tournament selection
- "League tables" → Fitness rankings within islands

Algorithm:
    1. Initialize K islands with N individuals each
    2. For each generation:
        a. **Local Improvement**: Apply ALNS to each individual in each island
        b. **Fitness Evaluation**: Compute objective values
        c. **Tournament Selection**: Replace weak individuals via k-tournament
        d. **Migration**: Periodically exchange best solutions between islands
        e. **Global Update**: Track global best across all islands

Complexity:
    - Time: O(T × K × N × (ALNS_cost + eval_cost))
    - Space: O(K × N × n) for island populations
    - Migration: O(K × migration_size) per interval

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023).
    "Volleyball premier league algorithm with ACO and ALNS for
    simultaneous pickup–delivery location routing problem."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from .params import MemeticIslandModelGAParams


class MemeticIslandModelGASolver(PolicyVizMixin):
    """
    Island Model Genetic Algorithm solver for VRPP.

    Maintains multiple independent sub-populations (islands) that evolve
    in parallel with periodic migration of best solutions.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticIslandModelGAParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Island Model GA solver.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: Island Model GA configuration parameters.
            mandatory_nodes: Nodes that must be visited.
            seed: Random seed for reproducibility.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Initialize constructive heuristic (ACO for solution generation)
        self.aco_solver = KSparseACOSolver(
            dist_matrix, wastes, capacity, R, C, params.aco_params, mandatory_nodes, seed=seed
        )
        self.constructor = self.aco_solver.constructor
        self.pheromone = self.aco_solver.pheromone

        # Initialize local search operator (ALNS for improvement)
        self.alns_solver = ALNSSolver(
            dist_matrix, wastes, capacity, R, C, params.alns_params, mandatory_nodes, seed=seed
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Island Model Genetic Algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        start_time = time.process_time()

        # Initialize islands (sub-populations)
        islands: List[List[Tuple[List[List[int]], float, float]]] = []
        for _ in range(self.params.n_islands):
            island_population = []
            for _ in range(self.params.island_size):
                routes = self.constructor.construct()
                routes = self._canonicalize_routes(routes)
                cost = self._calculate_cost(routes)
                revenue = sum(self.wastes.get(n, 0) * self.R for r in routes for n in r)
                profit = revenue - cost * self.C
                island_population.append((routes, profit, cost))
            islands.append(island_population)

        # Track global best solution across all islands
        best_routes, best_profit, best_cost = self._get_global_best(islands)

        # Evolution loop (generations)
        for generation in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # --- Phase 1: Local Improvement (per island) ---
            for island_idx in range(self.params.n_islands):
                new_island_population = []
                for routes, _profit, _cost in islands[island_idx]:
                    # Apply local search operator (ALNS)
                    improved_routes, improved_profit, improved_cost = self.alns_solver.solve(initial_solution=routes)
                    improved_routes = self._canonicalize_routes(improved_routes)
                    new_island_population.append((improved_routes, improved_profit, improved_cost))
                islands[island_idx] = new_island_population

            # --- Phase 2: Fitness Evaluation and Global Best Update ---
            gen_best_routes, gen_best_profit, gen_best_cost = self._get_global_best(islands)
            if gen_best_profit > best_profit:
                best_routes = copy.deepcopy(gen_best_routes)
                best_profit = gen_best_profit
                best_cost = gen_best_cost

            # --- Phase 3: Pheromone Update (Global Guidance) ---
            # Update ACO pheromones based on global best solution
            self._update_pheromones(best_routes, best_cost)

            # --- Phase 4: Tournament Selection and Replacement ---
            for island_idx in range(self.params.n_islands):
                islands[island_idx] = self._tournament_replacement(islands[island_idx])

            # --- Phase 5: Migration (Periodic) ---
            if generation > 0 and generation % self.params.migration_interval == 0:
                islands = self._migrate_best_solutions(islands)

            self._viz_record(
                iteration=generation,
                best_profit=best_profit,
                best_cost=best_cost,
                gen_best_profit=gen_best_profit,
                n_islands=self.params.n_islands,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Island Operations
    # ------------------------------------------------------------------

    def _tournament_replacement(
        self, island_population: List[Tuple[List[List[int]], float, float]]
    ) -> List[Tuple[List[List[int]], float, float]]:
        """
        Replace weakest individuals using tournament selection.

        Tournament Selection:
            1. Sort population by fitness (descending)
            2. Identify weakest k% individuals for replacement
            3. Generate new individuals via ACO constructor
            4. Replace weakest with new solutions

        Args:
            island_population: Current island population.

        Returns:
            Updated island population after replacement.

        Complexity: O(N log N) for sorting + O(k × construction_cost).
        """
        # Sort by fitness (profit descending, cost ascending, route hash for tie-breaking)
        island_population.sort(key=lambda x: (x[1], -x[2], self._hash_routes(x[0])), reverse=True)

        # Determine number of individuals to replace
        n_replace = int(self.params.island_size * self.params.replacement_rate)

        # Replace weakest individuals with new random solutions
        for i in range(self.params.island_size - n_replace, self.params.island_size):
            new_routes = self.constructor.construct()
            new_routes = self._canonicalize_routes(new_routes)
            new_cost = self._calculate_cost(new_routes)
            new_revenue = sum(self.wastes.get(n, 0) * self.R for r in new_routes for n in r)
            new_profit = new_revenue - new_cost * self.C
            island_population[i] = (new_routes, new_profit, new_cost)

        return island_population

    def _migrate_best_solutions(
        self, islands: List[List[Tuple[List[List[int]], float, float]]]
    ) -> List[List[Tuple[List[List[int]], float, float]]]:
        """
        Perform inter-island migration of best solutions.

        Migration Strategy (Ring Topology):
            Each island sends its best solution(s) to the next island.
            Island k sends to island (k+1) mod K.

        Args:
            islands: All island populations.

        Returns:
            Updated islands after migration.

        Complexity: O(K × migration_size).
        """
        # Extract best individual from each island
        migrants = []
        for island in islands:
            best_individual = max(island, key=lambda x: (x[1], -x[2], self._hash_routes(x[0])))
            migrants.append(best_individual)

        # Ring topology migration: Island i receives from island (i-1) mod K
        for i in range(self.params.n_islands):
            source_island_idx = (i - 1) % self.params.n_islands
            migrant = migrants[source_island_idx]

            # Replace a random individual (not the best) in target island
            if len(islands[i]) > 1:
                # Don't replace the best individual
                replacement_idx = self.rng.randint(1, len(islands[i]) - 1)
                islands[i][replacement_idx] = copy.deepcopy(migrant)

        return islands

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    def _get_global_best(
        self, islands: List[List[Tuple[List[List[int]], float, float]]]
    ) -> Tuple[List[List[int]], float, float]:
        """
        Find the global best solution across all islands.

        Args:
            islands: All island populations.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).

        Complexity: O(K × N) for scanning all individuals.
        """
        global_best = None
        for island in islands:
            island_best = max(island, key=lambda x: (x[1], -x[2], self._hash_routes(x[0])))
            if global_best is None or island_best[1] > global_best[1]:
                global_best = island_best

        return global_best if global_best else ([], 0.0, 0.0)

    def _canonicalize_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Canonicalize route representation by sorting routes.

        Ensures consistent ordering for hashing and comparison.

        Args:
            routes: List of routes.

        Returns:
            Sorted routes.

        Complexity: O(k log k) where k = number of routes.
        """
        return sorted([r for r in routes if r], key=lambda x: x[0] if x else 0)

    def _hash_routes(self, routes: List[List[int]]) -> str:
        """
        Compute deterministic hash for routes (for tie-breaking).

        Args:
            routes: List of routes.

        Returns:
            String hash representation.

        Complexity: O(n) for route traversal.
        """
        sorted_routes = self._canonicalize_routes(routes)
        return "|".join(",".join(map(str, r)) for r in sorted_routes)

    def _update_pheromones(self, routes: List[List[int]], cost: float) -> None:
        """
        Update ACO pheromones based on best solution (global guidance).

        Uses ACS-style pheromone update with evaporation and reinforcement.

        Args:
            routes: Best routes to reinforce.
            cost: Total cost of best routes.

        Complexity: O(n) for edge traversal.
        """
        if not routes or cost <= 0:
            return

        # Evaporate pheromones globally
        self.pheromone.evaporate_all(self.params.aco_params.rho)

        # Deposit pheromones on best solution edges
        delta_tau = 1.0 / cost if cost > 0 else 1.0
        for route in routes:
            if not route:
                continue
            self.pheromone.deposit_edge(0, route[0], delta_tau)
            for k in range(len(route) - 1):
                self.pheromone.deposit_edge(route[k], route[k + 1], delta_tau)
            self.pheromone.deposit_edge(route[-1], 0, delta_tau)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance traveled.

        Complexity: O(n) for route traversal.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
