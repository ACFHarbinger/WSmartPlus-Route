"""
Base acceptance criteria module.

Provides a common framework for single-point local search algorithms
using various move acceptance criteria (SA, GD, TA, etc.).
"""

import copy
import random
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.operators import (
    cluster_removal,
    greedy_insertion,
    random_removal,
    regret_2_insertion,
    worst_removal,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin


class BaseAcceptanceSolver(PolicyVizMixin):
    """
    Abstract base class for solvers using a move acceptance criterion.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        mandatory_nodes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(seed) if seed is not None else random.Random()

        # Default LLH pool
        self._llh_pool = [
            self._llh_random_greedy,
            self._llh_worst_regret,
            self._llh_cluster_greedy,
            self._llh_worst_greedy,
            self._llh_random_regret,
        ]

    @abstractmethod
    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """
        Determine if the candidate move should be accepted.
        """
        pass

    def _update_state(self, iteration: int):  # noqa: B027
        """
        Hook for updating internal solver state (e.g., temperature).
        """
        pass

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Standard iteration loop for acceptance-based metaheuristics.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initial solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)

        best_routes = copy.deepcopy(routes)
        best_profit = profit

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            self._update_state(iteration)

            # Select and apply a random LLH
            llh_idx = self.random.randint(0, self.params.n_llh - 1)
            llh = self._llh_pool[llh_idx]

            try:
                new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                new_profit = self._evaluate(new_routes)
            except Exception:
                continue

            # Acceptance check
            if self._accept(new_profit, profit, iteration):
                routes = new_routes
                profit = new_profit

                if profit > best_profit:
                    best_routes = copy.deepcopy(routes)
                    best_profit = profit

            self._record_telemetry(iteration, best_profit, profit)

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        """Record iteration data for visualization."""
        getattr(self, "_viz_record", lambda **k: None)(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
        )

    # --- LLH pool implementations ---

    def _llh_random_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = random_removal(routes, n, self.random)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_worst_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_cluster_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_worst_greedy(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = worst_removal(routes, n, self.dist_matrix)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_random_regret(self, routes: List[List[int]], n: int) -> List[List[int]]:
        partial, removed = random_removal(routes, n, self.random)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    # --- Common Helpers ---

    def _build_initial_solution(self) -> List[List[int]]:
        from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes

        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
