"""
Guided Indicators Hyper-Heuristic (GIHH) for VRPP.

GIHH is an adaptive hyper-heuristic that selects Low-Level Heuristics (LLHs)
based on their historical performance, guided by specific indicators
(e.g., improvement rate, time since last improvement).

Reference:
    - Akpinar, S. (2016). "A guided indicators hyper-heuristic for solving the
      vehicle routing problem with time windows."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..other.operators import random_removal
from ..other.operators.destroy.shaw import shaw_profit_removal, shaw_removal
from ..other.operators.destroy.string import string_removal
from ..other.operators.heuristics.greedy_initialization import build_greedy_routes
from ..other.operators.repair.greedy_blink import (
    greedy_insertion_with_blinks,
    greedy_profit_insertion_with_blinks,
)
from ..other.operators.repair.regret import regret_2_insertion, regret_2_profit_insertion
from .indicators import ImprovementRateIndicator, TimeBasedIndicator
from .params import GIHHParams
from .solution import Solution


class GIHHSolver:
    """
    GIHH solver implementation for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GIHHParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random(42)

        # Initialise indicators
        self.indicators = [
            ImprovementRateIndicator(),
            TimeBasedIndicator(),
        ]

        # LLH pool (destroy and repair combinations)
        # For simplicity, we define them implicitly in _apply_operator

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Runs the GIHH metaheuristic."""
        if len(self.dist_matrix) <= 1:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # 1. Initial Solution
        initial_routes = build_greedy_routes(
            nodes=list(range(1, len(self.dist_matrix))),
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )

        current_sol = Solution(initial_routes, self._evaluate(initial_routes))
        best_sol = copy.deepcopy(current_sol)

        iteration = 0
        while True:
            # Check stopping criteria
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break
            if iteration >= self.params.max_iterations:
                break

            # 2. Select and Apply LLH
            candidate_sol = self._apply_operator(current_sol, iteration)

            # 3. Update Indicators
            delta = candidate_sol.profit - current_sol.profit
            for indicator in self.indicators:
                indicator.update(delta, elapsed)

            # 4. Acceptance (Greedy)
            if candidate_sol.profit >= current_sol.profit:
                current_sol = candidate_sol
                if current_sol.profit > best_sol.profit:
                    best_sol = copy.deepcopy(current_sol)

            iteration += 1

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_sol.profit,
                current_profit=current_sol.profit,
                elapsed=elapsed,
            )

        best_cost = self._cost(best_sol.routes)
        return best_sol.routes, best_sol.profit, best_cost

    def _apply_operator(self, current: Solution, iteration: int) -> Solution:
        """
        Selects and applies a ruin-and-recreate operator based on indicators.
        In this simplified GIHH, we pick destroy/repair operators.
        """
        candidate = copy.deepcopy(current)
        all_nodes = [n for r in candidate.routes for n in r]
        if not all_nodes:
            return candidate

        # 1. Ruin (Destroy)
        # Choose destroy operator based on indicators (simplified: random choice)
        op_type = self.rng.random()
        n_remove = max(1, min(len(all_nodes) // 4, 10))
        expand_pool = self.params.vrpp

        if op_type < 0.33:
            # Shaw removal
            if self.params.profit_aware_operators:
                candidate.routes, _ = shaw_profit_removal(
                    candidate.routes, n_remove, self.dist_matrix, self.wastes, self.R, self.C, rng=self.rng
                )
            else:
                candidate.routes, _ = shaw_removal(candidate.routes, n_remove, self.dist_matrix)
        elif op_type < 0.66:
            # String removal
            candidate.routes, _ = string_removal(candidate.routes, n_remove, self.dist_matrix, rng=self.rng)
        else:
            # Random removal
            candidate.routes, _ = random_removal(candidate.routes, n_remove, self.rng)

        # 2. Recreate (Re-insertion)
        # Randomly choose between blink and regret-2
        if self.rng.random() < 0.5:
            if self.params.profit_aware_operators:
                candidate.routes = greedy_profit_insertion_with_blinks(
                    candidate.routes,
                    list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                candidate.routes = greedy_insertion_with_blinks(
                    candidate.routes,
                    list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
        else:
            if self.params.profit_aware_operators:
                candidate.routes = regret_2_profit_insertion(
                    candidate.routes,
                    list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                candidate.routes = regret_2_insertion(
                    candidate.routes,
                    list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )

        candidate.profit = self._evaluate(candidate.routes)
        return candidate

    def _apply_perturbation_operator(self, current: Solution) -> Solution:
        """Applies a stronger ruin-and-recreate for stagnation escape."""
        candidate = copy.deepcopy(current)
        all_nodes = [n for r in candidate.routes for n in r]
        if not all_nodes:
            return candidate

        n_remove = max(1, min(len(all_nodes) // 2, 20))
        expand_pool = self.params.vrpp

        # Use random removal for diversity
        candidate.routes, _ = random_removal(candidate.routes, n_remove, self.rng)

        # Re-insert with profit-aware regret-2 if enabled
        if self.params.profit_aware_operators:
            candidate.routes = regret_2_profit_insertion(
                candidate.routes,
                list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            candidate.routes = regret_2_insertion(
                candidate.routes,
                list(set(range(1, len(self.dist_matrix))) - set(all_nodes)),
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

        candidate.profit = self._evaluate(candidate.routes)
        return candidate

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculates Net Profit ($)."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates total distance (km)."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i]][route[i + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
