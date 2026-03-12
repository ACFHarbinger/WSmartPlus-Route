"""
Hybrid Volleyball Premier League (HVPL) Solver.

Combines Ant Colony Optimization (ACO) for construction and global guidance
with Adaptive Large Neighborhood Search (ALNS) for local improvement (Coaching),
within a population-based framework (Leagues).

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023).
    "Volleyball premier league algorithm with ACO and ALNS for
    simultaneous pickup–delivery location routing problem."
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from .params import HVPLParams


class HVPLSolver(PolicyVizMixin):
    """
    Hybrid Volleyball Premier League solver for VRP variants.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        # Initialize ACO components for constructor and pheromones
        # We reuse KSparseACOSolver initialization logic
        self.aco_internal = KSparseACOSolver(
            dist_matrix, wastes, capacity, R, C, params.aco_params, mandatory_nodes, seed=seed
        )
        self.pheromone = self.aco_internal.pheromone
        self.constructor = self.aco_internal.constructor

        # Initialize ALNS solver for the "Coaching" phase
        self.coaching_solver = ALNSSolver(
            dist_matrix, wastes, capacity, R, C, params.alns_params, mandatory_nodes, seed=seed
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the HVPL algorithm.
        """
        start_time = time.process_time()

        # 1. Initialization: Create the initial population (Teams)
        population: List[Tuple[List[List[int]], float, float]] = []
        for _i in range(self.params.n_teams):
            routes = self.constructor.construct()
            routes = self._canonicalize_routes(routes)
            cost = self._calculate_cost(routes)
            # Profit calculation
            rev = sum(self.wastes.get(n, 0) * self.R for r in routes for n in r)
            profit = rev - cost * self.C
            population.append((routes, profit, cost))

        best_routes, best_profit, best_cost = self._get_best(population)

        # 2. League Season Iterations
        for _iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # 3. Coaching Phase: Apply ALNS to each team
            new_population = []
            for _i, (routes, _profit, _cost) in enumerate(population):
                # Coaching session (ALNS solve)
                c_routes, c_profit, c_cost = self.coaching_solver.solve(initial_solution=routes)
                c_routes = self._canonicalize_routes(c_routes)
                new_population.append((c_routes, c_profit, c_cost))

            population = new_population

            # 4. Global Competition: Update best-so-far
            iter_best_routes, iter_best_profit, iter_best_cost = self._get_best(population)
            if iter_best_profit > best_profit:
                best_routes = copy.deepcopy(iter_best_routes)
                best_profit = iter_best_profit
                best_cost = iter_best_cost

            # 5. Pheromone Update: Global guidance
            # Deposit pheromones on the best team's edges
            self._update_pheromones(best_routes, best_cost)

            self._viz_record(
                iteration=_iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iter_best_profit,
                population_size=len(population),
            )

            # 6. Substitution Phase: Replace weakest teams
            population.sort(key=lambda x: (x[1], -x[2], self._hash_routes(x[0])), reverse=True)
            n_sub = int(self.params.n_teams * self.params.sub_rate)

            for i in range(self.params.n_teams - n_sub, self.params.n_teams):
                # Replace with a new solution generated with updated pheromones
                s_routes = self.constructor.construct()
                s_routes = self._canonicalize_routes(s_routes)
                s_cost = self._calculate_cost(s_routes)
                s_rev = sum(self.wastes.get(n, 0) * self.R for r in s_routes for n in r)
                s_profit = s_rev - s_cost * self.C
                population[i] = (s_routes, s_profit, s_cost)

        return best_routes, best_profit, best_cost

    def _canonicalize_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Sort routes by their first node to ensure consistent ordering."""
        return sorted([r for r in routes if r], key=lambda x: x[0] if x else 0)

    def _hash_routes(self, routes: List[List[int]]) -> str:
        # Sort routes by their first node to ensure consistent ordering for hashing
        sorted_routes = sorted([r for r in routes if r], key=lambda x: x[0] if x else 0)
        return "|".join(",".join(map(str, r)) for r in sorted_routes)

    def _get_best(self, population: List[Tuple[List[List[int]], float, float]]) -> Tuple[List[List[int]], float, float]:
        """Get the highest-profit solution from the population with deterministic tie-break."""
        # Use (profit, negative_cost, hash) for deterministic tie-breaking.
        return max(population, key=lambda x: (x[1], -x[2], self._hash_routes(x[0])))

    def _update_pheromones(self, routes: List[List[int]], cost: float) -> None:
        """ACS style global pheromone update."""
        if not routes or cost <= 0:
            return

        # Evaporate
        self.pheromone.evaporate_all(self.params.aco_params.rho)

        # Deposit
        delta = self.params.aco_params.elitist_weight / cost
        for route in routes:
            if not route:
                continue
            self.pheromone.update_edge(0, route[0], delta, evaporate=False)
            for k in range(len(route) - 1):
                self.pheromone.update_edge(route[k], route[k + 1], delta, evaporate=False)
            self.pheromone.update_edge(route[-1], 0, delta, evaporate=False)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
