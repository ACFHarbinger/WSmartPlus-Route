"""
Reinforcement Learning + Great Deluge Hyper-Heuristic (RL-GD-HH) for VRPP.

This solver implements the adaptive selection of Low-Level Heuristics (LLHs)
using a Reinforcement Learning (RL) mechanism paired with a Great Deluge (GD)
acceptance criterion.

Key components:
1.  **RL-Based Selection**: Maintains a utility/score for each LLH, updated
    based on its performance (reward) in terms of solution improvement.
2.  **Great Deluge Acceptance**: Accept any candidate solution whose quality
    (profit) is above a linearly increasing water level.
3.  **LLH Pool**: A diverse set of ruin-and-recreate operators.

References:
    - Ozcan, E., et al. (2010). "Reinforcement learning-great deluge
      hyper-heuristic for solving examination timetabling problems."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..other.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
)
from .params import RLGDHHParams


class RLGDHHSolver:
    """
    RL-GD-HH solver implementation for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RLGDHHParams,
        mandatory_nodes: Optional[List[int]] = None,
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)

        # Initialise LLH utilities (scores)
        self.utilities = [self.params.initial_utility] * 4
        self._llh_pool = [
            self._llh_relocate,
            self._llh_shaw,
            self._llh_string,
            self._llh_regret2,
        ]

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Runs the RL-GD-HH metaheuristic."""
        if not self.nodes:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Initial solution
        current_routes = self._initialize_solution()
        current_profit = self._evaluate(current_routes)

        best_routes = copy.deepcopy(current_routes)
        best_profit = current_profit

        # water level (initialised based on initial profit)
        water_level = current_profit * (1.0 - self.params.flood_margin) if current_profit > 0 else -100.0

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break

            # Selection Phase: Select LLH based on max utility (Ozcan et al. 2010)
            # or epsilon-greedy/tournament
            llh_idx = self._select_llh()
            llh = self._llh_pool[llh_idx]

            # Application Phase
            new_routes = llh(copy.deepcopy(current_routes))
            new_profit = self._evaluate(new_routes)

            # Acceptance Phase: Great Deluge (Maximisation)
            accepted = new_profit >= water_level

            # Adaptation Phase (Reward RL)
            reward = 0.0
            if new_profit > current_profit:
                reward = self.params.reward_improvement
            elif new_profit == current_profit:
                reward = self.params.reward_neutral
            else:
                reward = self.params.penalty_worsening

            self.utilities[llh_idx] = max(
                self.params.utility_upper_bound, min(self.params.min_utility, self.utilities[llh_idx] + reward)
            )

            if accepted:
                current_routes = new_routes
                current_profit = new_profit

                if current_profit > best_profit:
                    best_routes = copy.deepcopy(current_routes)
                    best_profit = current_profit

            # Update water level (flood rises linearly)
            water_level += self.params.rain_speed * abs(best_profit + 1)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                water_level=water_level,
                selected_llh=llh_idx,
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    def _select_llh(self) -> int:
        """Selects LLH index using max-utility with ties broken randomly."""
        max_util = max(self.utilities)
        candidates = [i for i, u in enumerate(self.utilities) if u == max_util]
        return self.random.choice(candidates)

    def _llh_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """Simple node relocation (1-0 exchange)."""
        if not routes:
            return routes
        new_routes = copy.deepcopy(routes)
        ridx = self.random.randint(0, len(new_routes) - 1)
        if not new_routes[ridx]:
            return new_routes

        n_idx = self.random.randint(0, len(new_routes[ridx]) - 1)
        node = new_routes[ridx].pop(n_idx)

        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            new_routes = greedy_profit_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            new_routes = greedy_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        return new_routes

    def _llh_shaw(self, routes: List[List[int]]) -> List[List[int]]:
        """Shaw removal + regret-2 insertion."""
        n = max(1, min(len(self.nodes) // 4, 10))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            partial, removed = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            partial, removed = shaw_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _llh_string(self, routes: List[List[int]]) -> List[List[int]]:
        """String removal + greedy insertion."""
        n = max(1, min(len(self.nodes) // 4, 10))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        partial, removed = string_removal(routes, n, self.dist_matrix, rng=self.random)

        if use_profit:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _llh_regret2(self, routes: List[List[int]]) -> List[List[int]]:
        """Random removal + regret-2 insertion."""
        n = max(1, min(len(self.nodes) // 5, 5))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        partial, removed = random_removal(routes, n, rng=self.random)
        if use_profit:
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    # ------------------------------------------------------------------
    # Functional Helpers
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """Randomized greedy construction of a starting feasible routing."""
        nodes = copy.copy(self.nodes)
        self.random.shuffle(nodes)
        self.random.shuffle(nodes)

        try:
            return greedy_profit_insertion(
                [],
                nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
            )
        except Exception:
            return [[n] for n in nodes]

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculates fitness F(g) as Net Profit ($)."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates distance-based travel cost (km)."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i]][route[i + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
