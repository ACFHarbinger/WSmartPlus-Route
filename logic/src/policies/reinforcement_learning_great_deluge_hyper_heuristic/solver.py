"""
RL-GD-HH Solver for VRPP.

This solver implements the Reinforcement Learning – Great Deluge Hyper-Heuristic
framework. It utilizes online learning to select the most effective
neighborhood-search operators (Low-Level Heuristics) and a threshold-based
Great Deluge algorithm to manage move acceptance across the search trajectory.

Reference:
    Ozcan, E., Misir, M., Ochoa, G., & Burke, E. K. (2010).
    "A Reinforcement Learning – Great-Deluge Hyper-heuristic for Examination Timetabling".
    Bibliography: bibliography/Reinforcement_Learning_Great_Deluge_Hyper-Heuristic.pdf
"""

import contextlib
import copy
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..other.operators import (
    greedy_insertion,
    regret_2_insertion,
    shaw_removal,
    string_removal,
)
from .params import RLGDHHParams


class RLGDHHSolver:
    """
    Reinforcement Learning Great Deluge Hyper-Heuristic Solver.

    The solver coordinates multiple independent search operators (LLHs)
    using a Reinforcement Learning scheme (RL) that assigns utilities
    based on success. Solution eligibility is determined by the "Water Level"
    of the Great Deluge mechanism, which provides a deterministic,
    time-dependent threshold for escaping local optima.

    Algorithm Structure:
    - RL: 'Max' selection strategy based on success-driven utility.
    - GD: Time-based linear level growth (Section 3.2 of the paper).
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
        seed: Optional[int] = None,
    ):
        """
        Initialize the RL-GD-HH solver.

        Args:
            dist_matrix: Pairwise node distances.
            wastes: Node profits ($).
            capacity: Vehicle capacity constraint.
            R: Revenue coefficient ($/unit).
            C: Cost coefficient ($/km).
            params: Parameters for RL selection and GD acceptance.
            mandatory_nodes: Nodes that must appear in even feasible routes.
            seed: RNG seed for reproducible execution.
        """
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
        self.start_time = time.process_time()

        # Define our pool of Low-Level Heuristics (LLHs)
        self.heuristics: List[Callable] = [
            self._llh_swap,
            self._llh_relocate,
            self._llh_2opt,
            self._llh_shaw,
            self._llh_string,
            self._llh_regret2,
        ]

        # 1. Optimistic Initialization (Paper p. 16: "set all utilities to 0.75 * UB")
        # This encourages early exploration of all operators.
        init_val = 0.75 * self.params.utility_upper_bound
        self.utilities = [init_val for _ in self.heuristics]

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Executive main loop for the RL-GD-HH algorithm.

        Follows Figure 2 (Step-by-step logic) of Ozcan et al. (2010).

        Algorithm Process:
        1.  Initialize: Create initial feasible solution and calculate f0.
        2.  Select: Choose heuristic with max utility ('Max' strategy).
        3.  Apply & Evaluate: Transform current solution and measure fitness.
        4.  Accept/Reject (GD): Compare candidate fitness against time-based level.
        5.  Reinforce (RL): Increment/Decrement utility based on move quality.

        Returns:
            Tuple[List[List[int]], float]:
                - best_solution: The most profitable routing found.
                - best_fitness: The net profit score.
        """
        # Phase 1: Initial state construction
        current_solution = self._initialize_solution()
        current_fitness = self._evaluate(current_solution)
        f0 = current_fitness  # Benchmark fitness for slope calculation

        best_solution = copy.deepcopy(current_solution)
        best_fitness = current_fitness

        # Estimate the target fitness (ideal end-of-search objective)
        target_f = f0 * self.params.target_fitness_multiplier

        # Main search loop
        for iteration in range(self.params.max_iterations):
            elapsed = time.process_time() - self.start_time
            if elapsed > self.params.time_limit:
                break

            # 2. Heuristic Selection (Paper Section 3.2: Max Strategy)
            # Picks the 'best' performing operator currently in memory.
            chosen_idx = self._select_heuristic()
            heuristic = self.heuristics[chosen_idx]

            # Exploratory action
            candidate_solution = heuristic(copy.deepcopy(current_solution))
            candidate_fitness = self._evaluate(candidate_solution)

            # 3. Dynamic Move Acceptance (Great Deluge Figure 2, Step 19)
            # Level(t) = f0 + (TargetF - f0) * (t/T)
            # Ozcan et al. (2010) Step 19-20: If f(S') >= f(S) OR f(S') >= Level, accept.
            progress = min(1.0, elapsed / self.params.time_limit)
            water_level = f0 + (target_f - f0) * progress

            if candidate_fitness >= current_fitness:
                # Improving or Equal Move: Always Accept (Paper Step 14)
                current_solution = candidate_solution
                current_fitness = candidate_fitness

                # RL Reward (Step 15: "u_i = reward(u_i)")
                self.utilities[chosen_idx] = min(
                    self.params.utility_upper_bound,
                    self.utilities[chosen_idx] + self.params.reward_improvement,
                )

                # Update Global Best
                if current_fitness > best_fitness:
                    best_solution = copy.deepcopy(current_solution)
                    best_fitness = current_fitness
            elif candidate_fitness >= water_level:
                # Worsening but Acceptable Move (Paper Step 19-20)
                current_solution = candidate_solution
                current_fitness = candidate_fitness

                # RL Penalty (Step 18: "u_i = punish(u_i)")
                self.utilities[chosen_idx] = max(
                    self.params.min_utility,
                    self.utilities[chosen_idx] - self.params.penalty_worsening,
                )
            else:
                # Move Rejected (Below Water Level)
                # RL Penalty (Step 18: "u_i = punish(u_i)")
                self.utilities[chosen_idx] = max(
                    self.params.min_utility,
                    self.utilities[chosen_idx] - self.params.penalty_worsening,
                )

            # Telemetry for visualization
            if iteration % 10 == 0:
                getattr(self, "_viz_record", lambda **k: None)(
                    iteration=iteration,
                    best_profit=best_fitness,
                    best_cost=self._cost(best_solution),
                )

        return best_solution, best_fitness

    # ------------------------------------------------------------------
    # Heuristic Selection Strategy
    # ------------------------------------------------------------------

    def _select_heuristic(self) -> int:
        """
        Selects an LLH using the 'Max' strategy.

        Paper Reference (p. 16):
        "using maximal utility value to select a heuristic... reported to
        outperform [roulette wheel] selection."
        """
        max_val = max(self.utilities)
        indices = [i for i, val in enumerate(self.utilities) if val == max_val]
        return self.random.choice(indices)

    # ------------------------------------------------------------------
    # Low-Level Heuristics (LLHs)
    # ------------------------------------------------------------------

    def _llh_swap(self, routes: List[List[int]]) -> List[List[int]]:
        """Inter/Intra-route exchange of two randomly selected nodes."""
        flat = [n for r in routes for n in r]
        if len(flat) < 2:
            return routes

        n1, n2 = self.random.sample(flat, 2)
        new_routes = []
        for route in routes:
            new_r = [n2 if n == n1 else n1 if n == n2 else n for n in route]
            new_routes.append(new_r)
        return new_routes

    def _llh_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """Random node relocation using greedy insertion feasibility check."""
        flat = [n for r in routes for n in r]
        if not flat:
            return routes

        node = self.random.choice(flat)
        new_routes = [[n for n in r if n != node] for r in routes]
        new_routes = [r for r in new_routes if r]

        with contextlib.suppress(Exception):
            new_routes = greedy_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        return new_routes

    def _llh_2opt(self, routes: List[List[int]]) -> List[List[int]]:
        """Geometric-based route inversion to untangle crossed edges."""
        if not routes:
            return routes
        ridx = self.random.randint(0, len(routes) - 1)
        route = routes[ridx]

        if len(route) >= 3:
            i, j = sorted(self.random.sample(range(len(route)), 2))
            routes[ridx] = route[:i] + route[i:j][::-1] + route[j:]

        return routes

    def _llh_shaw(self, routes: List[List[int]]) -> List[List[int]]:
        """Shaw removal + regret-2 insertion."""
        n = max(1, min(len(self.nodes) // 4, 10))
        partial, removed = shaw_removal(routes, n, self.dist_matrix)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_string(self, routes: List[List[int]]) -> List[List[int]]:
        """String removal + greedy insertion."""
        n = max(1, min(len(self.nodes) // 4, 10))
        partial, removed = string_removal(routes, n, self.dist_matrix, rng=self.random)
        return greedy_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _llh_regret2(self, routes: List[List[int]]) -> List[List[int]]:
        """Random removal + regret-2 insertion."""
        n = max(1, min(len(self.nodes) // 5, 5))
        from ..other.operators import random_removal

        partial, removed = random_removal(routes, n, rng=self.random)
        return regret_2_insertion(
            partial,
            removed,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            mandatory_nodes=self.mandatory_nodes,
        )

    # ------------------------------------------------------------------
    # Functional Helpers
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """Randomized greedy construction of a starting feasible routing."""
        nodes = copy.copy(self.nodes)
        self.random.shuffle(nodes)
        with contextlib.suppress(Exception):
            return greedy_insertion(
                [],
                nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
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
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
