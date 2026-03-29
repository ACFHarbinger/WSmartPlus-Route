"""
Guided Indicators Hyper-Heuristic (GIHH) for VRPP.

GIHH is an adaptive hyper-heuristic that selects Low-Level Heuristics (LLHs)
based on their historical performance, guided by two specific indicators:
1. Improvement Rate Indicator (IRI): Measures solution quality improvement
2. Time-based Indicator (TBI): Measures computational efficiency

The algorithm uses epsilon-greedy selection to balance exploration and exploitation,
and employs a hill climbing acceptance criterion to escape local optima.

References:
    - Chen, B., Qu, R., Bai, R., & Laesanklang, W. (2018). "A hyper-heuristic with
      two guidance indicators for bi-objective mixed-shift vehicle routing problem
      with time windows." European Journal of Operational Research, 269(2), 661-675.
    - Akpinar, S. (2016). "Hybrid large neighbourhood search algorithm for capacitated
      vehicle routing problem." Expert Systems with Applications, 61, 28-38.
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

    This hyper-heuristic uses two guidance indicators (IRI and TBI) to adaptively
    select Low-Level Heuristics based on their historical performance. It employs
    epsilon-greedy selection for exploration-exploitation balance and hill climbing
    acceptance to escape local optima.
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

        # Initialize guidance indicators with per-operator tracking
        self.iri = ImprovementRateIndicator(window_size=params.iri_window)
        self.tbi = TimeBasedIndicator(window_size=params.tbi_window)

        # Define Low-Level Heuristics (LLH) pool as operator combinations
        # Each operator is a tuple of (destroy_type, repair_type)
        self.operators = [
            ("shaw", "blink"),
            ("shaw", "regret2"),
            ("string", "blink"),
            ("string", "regret2"),
            ("random", "blink"),
            ("random", "regret2"),
        ]

        # Adaptive parameters
        self.epsilon = params.epsilon
        self.accept_worse_prob = params.accept_worse_prob

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Runs the GIHH metaheuristic with guided operator selection.

        Algorithm:
            1. Generate initial solution
            2. While stopping criteria not met:
                a. Select operator using epsilon-greedy strategy
                b. Apply selected operator and measure performance
                c. Update guidance indicators (IRI and TBI)
                d. Accept/reject solution using hill climbing criterion
                e. Decay epsilon and acceptance probability
            3. Return best solution found

        Returns:
            Tuple of (routes, profit, cost) for the best solution.
        """
        if len(self.dist_matrix) <= 1:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # 1. Generate Initial Solution
        initial_routes = build_greedy_routes(
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )

        current_sol = Solution(
            initial_routes,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
        )
        best_sol = copy.deepcopy(current_sol)

        iteration = 0
        while True:
            # Check stopping criteria
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break
            if iteration >= self.params.max_iterations:
                break

            # 2. Select operator using epsilon-greedy strategy
            selected_operator = self._select_operator()

            # 3. Apply selected operator and measure performance
            operator_start_time = time.process_time()
            candidate_sol = self._apply_selected_operator(current_sol, selected_operator)
            operator_elapsed = time.process_time() - operator_start_time

            # 4. Update guidance indicators with operator-specific metrics
            improvement = candidate_sol.profit - current_sol.profit
            operator_name = f"{selected_operator[0]}_{selected_operator[1]}"
            self.iri.update(operator_name, improvement)
            self.tbi.update(operator_name, operator_elapsed)

            # 5. Acceptance using Hill Climbing criterion
            accept = False
            if candidate_sol.profit > current_sol.profit:
                # Always accept improvements
                accept = True
            elif self.params.accept_equal and candidate_sol.profit == current_sol.profit:
                # Accept equal solutions if configured
                accept = True
            elif self.rng.random() < self.accept_worse_prob:
                # Accept worse solutions with probability (hill climbing)
                accept = True

            if accept:
                current_sol = candidate_sol
                if current_sol.profit > best_sol.profit:
                    best_sol = copy.deepcopy(current_sol)

            # 6. Decay adaptive parameters
            self.epsilon = max(self.params.min_epsilon, self.epsilon * self.params.epsilon_decay)
            self.accept_worse_prob *= self.params.acceptance_decay

            iteration += 1

            # Optional visualization callback
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_sol.profit,
                current_profit=current_sol.profit,
                elapsed=elapsed,
            )

        best_cost = self._cost(best_sol.routes)
        return best_sol.routes, best_sol.profit, best_cost

    def _select_operator(self) -> Tuple[str, str]:
        """
        Select operator using epsilon-greedy strategy based on guidance indicators.

        Implements the selection mechanism from Chen et al. (2018):
            Score_i = (w_IRI * IRI_i) + (w_TBI * TBI_i)

        With probability epsilon, selects a random operator (exploration).
        Otherwise, selects the operator with the highest weighted score (exploitation).

        Returns:
            Tuple of (destroy_type, repair_type) representing the selected operator.
        """
        # Epsilon-greedy: Explore with probability epsilon
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.operators)

        # Collect raw IRI and TBI scores for all operators
        iri_scores = []
        tbi_scores = []
        for operator in self.operators:
            operator_name = f"{operator[0]}_{operator[1]}"
            iri_scores.append(self.iri.get_score(operator_name))
            tbi_scores.append(self.tbi.get_score(operator_name))

        # Calculate minimum and maximum values
        min_iri, max_iri = min(iri_scores), max(iri_scores)
        min_tbi, max_tbi = min(tbi_scores), max(tbi_scores)

        # Exploitation: Select operator with highest weighted score
        best_operator = self.operators[0]
        best_score = float("-inf")

        for i, operator in enumerate(self.operators):
            # Min-Max normalization
            norm_iri = (iri_scores[i] - min_iri) / (max_iri - min_iri + 1e-9)
            norm_tbi = (tbi_scores[i] - min_tbi) / (max_tbi - min_tbi + 1e-9)

            # Calculate weighted score using normalized values
            weighted_score = (self.params.iri_weight * norm_iri) + (self.params.tbi_weight * norm_tbi)

            if weighted_score > best_score:
                best_score = weighted_score
                best_operator = operator

        return best_operator

    def _apply_selected_operator(self, current: Solution, operator: Tuple[str, str]) -> Solution:
        """
        Apply the selected destroy-repair operator combination to the current solution.

        Args:
            current: Current solution to modify.
            operator: Tuple of (destroy_type, repair_type).

        Returns:
            Modified candidate solution.
        """
        destroy_type, repair_type = operator
        candidate = copy.deepcopy(current)

        all_nodes = [n for r in candidate.routes for n in r]
        if not all_nodes:
            return candidate

        # Determine removal size
        n_remove = max(1, min(len(all_nodes) // 4, 10))
        expand_pool = self.params.vrpp

        # 1. Apply Destroy Operator
        if destroy_type == "shaw":
            if self.params.profit_aware_operators:
                candidate.routes, removed_nodes = shaw_profit_removal(
                    candidate.routes, n_remove, self.dist_matrix, self.wastes, self.R, self.C, rng=self.rng
                )
            else:
                candidate.routes, removed_nodes = shaw_removal(
                    candidate.routes, n_remove, self.dist_matrix, rng=self.rng
                )
        elif destroy_type == "string":
            candidate.routes, removed_nodes = string_removal(candidate.routes, n_remove, self.dist_matrix, rng=self.rng)
        elif destroy_type == "random":
            candidate.routes, removed_nodes = random_removal(candidate.routes, n_remove, self.rng)

        # Localize reinsertion pool based on removed nodes
        nodes_to_insert_set = set(removed_nodes)
        current_nodes = set(n for r in candidate.routes for n in r)

        for node in removed_nodes:
            distances = self.dist_matrix[node]
            nearest_indices = np.argsort(distances)

            count = 0
            for neighbor_idx in nearest_indices:
                neighbor = int(neighbor_idx)
                if neighbor == 0 or neighbor == node:
                    continue

                if neighbor not in current_nodes:
                    nodes_to_insert_set.add(neighbor)

                count += 1
                if count >= 15:
                    break

        nodes_to_insert = list(nodes_to_insert_set)

        # 2. Apply Repair Operator
        if repair_type == "blink":
            if self.params.profit_aware_operators:
                candidate.routes = greedy_profit_insertion_with_blinks(
                    candidate.routes,
                    nodes_to_insert,
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
                    nodes_to_insert,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
        elif repair_type == "regret2":
            if self.params.profit_aware_operators:
                candidate.routes = regret_2_profit_insertion(
                    candidate.routes,
                    nodes_to_insert,
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
                    nodes_to_insert,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )

        # Evaluate candidate solution
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
