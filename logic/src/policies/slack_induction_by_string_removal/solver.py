"""
SISR Solver Module.

This module implements the Slack Induction by String Removal (SISR) metaheuristic.
It manages the iterative process of string removal (ruin) and greedy insertion
with blinks (recreate), accepted via Simulated Annealing.

Attributes:
    None

Example:
    >>> from logic.src.policies.slack_induction_by_string_removal.solver import SISRSolver
    >>> solver = SISRSolver(dist_matrix, demands, ...)
    >>> result = solver.solve()
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..operators.destroy_operators import string_removal
from ..operators.repair_operators import greedy_insertion_with_blinks
from .params import SISRParams


class SISRSolver:
    """
    Solver implementing the SISR metaheuristic.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SISRParams,
    ):
        """
        Initialize SISR Solver.

        Args:
            params: Parameters for the SISR algorithm.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run the SISR algorithm.
        """
        if initial_solution:
            current_routes = [r[:] for r in initial_solution]
        else:
            current_routes = self._build_initial_solution()

        best_routes = [r[:] for r in current_routes]
        best_cost = self._calculate_cost(best_routes)
        current_cost = best_cost

        T = self.params.start_temp
        start_time = time.time()

        n_nodes = len(self.dist_matrix) - 1
        n_remove = max(1, int(n_nodes * self.params.destroy_ratio))

        for it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # SISR Iteration
            # 1. Destroy
            partial_routes, removed = string_removal(
                copy.deepcopy(current_routes),
                n_remove,
                self.dist_matrix,
                max_string_len=self.params.max_string_len,
                avg_string_len=self.params.avg_string_len,
            )

            # 2. Repair
            new_routes = greedy_insertion_with_blinks(
                partial_routes,
                removed,
                self.dist_matrix,
                self.demands,
                self.capacity,
                blink_rate=self.params.blink_rate,
            )

            new_cost = self._calculate_cost(new_routes)

            # Acceptance (Simulated Annealing)
            delta = new_cost - current_cost
            accept = False

            if delta < 1e-6:
                accept = True
            else:
                prob = math.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    accept = True

            if accept:
                current_routes = new_routes
                current_cost = new_cost
                if current_cost < best_cost - 1e-6:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost

            # Cooling
            T *= self.params.cooling_rate

        collected_revenue = sum(self.demands.get(node, 0) * self.R for route in best_routes for node in route)
        profit = collected_revenue - (best_cost * self.C)

        return best_routes, profit, best_cost

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += float(self.dist_matrix[0, route[0]])
            for i in range(len(route) - 1):
                total += float(self.dist_matrix[route[i], route[i + 1]])
            total += float(self.dist_matrix[route[-1], 0])
        return total

    def _build_initial_solution(self) -> List[List[int]]:
        """Greedy constructive heuristic."""
        nodes = list(self.demands.keys())
        random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        for node in nodes:
            demand = self.demands.get(node, 0.0)
            if load + demand <= self.capacity:
                curr_route.append(node)
                load += demand
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = demand
        if curr_route:
            routes.append(curr_route)
        return routes
