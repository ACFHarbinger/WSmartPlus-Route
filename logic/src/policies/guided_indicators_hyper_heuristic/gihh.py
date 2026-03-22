"""
GIHH (Hyper-Heuristic with Two Guidance Indicators) implementation.

A selection hyper-heuristic that uses two complementary indicators to guide
the adaptive selection of low-level heuristics during local search.

Reference:
    Chen, B., Qu, R., Bai, R., & Laesanklang, W.,
    "A hyper-heuristic with two guidance indicators for bi-objective
    mixed-shift vehicle routing problem with time windows", 2018
"""

import random
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..other.operators.destroy.shaw import shaw_removal
from ..other.operators.destroy.string import string_removal
from ..other.operators.heuristics.greedy_initialization import build_greedy_routes
from ..other.operators.repair.greedy_blink import greedy_insertion_with_blinks
from ..other.operators.repair.regret import regret_2_insertion
from .indicators import ImprovementRateIndicator, TimeBasedIndicator
from .params import GIHHParams
from .solution import Solution


class GIHHSolver:
    """
    Hyper-Heuristic with Two Guidance Indicators for VRPP.

    Uses Improvement Rate Indicator (IRI) and Time-based Indicator (TBI)
    to adaptively select low-level heuristics during search.
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
        seed: Optional[int] = None,
    ):
        """
        Initialize the GIHH solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: GIHH parameters.
            mandatory_nodes: List of local node indices that MUST be visited.
            seed: Random seed for reproducibility.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.rng = random.Random(seed) if seed is not None else random.Random(42)

        self.n_nodes = len(dist_matrix) - 1

        # Guidance indicators
        self.iri = ImprovementRateIndicator(window_size=params.iri_window)
        self.tbi = TimeBasedIndicator(window_size=params.tbi_window)

        # Operator performance tracking
        self.operator_scores: Dict[str, float] = {}
        self.operator_times: Dict[str, Deque[float]] = {}
        self.operator_improvements: Dict[str, Deque[float]] = {}

        # Initialize tracking for all operators
        all_operators = params.move_operators + params.perturbation_operators
        for op in all_operators:
            self.operator_scores[op] = 1.0
            self.operator_times[op] = deque(maxlen=params.memory_size)
            self.operator_improvements[op] = deque(maxlen=params.memory_size)

        # Acceptance parameters
        self.accept_worse_prob = params.accept_worse_prob

        # Exploration parameter
        self.epsilon = params.epsilon

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the GIHH algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        best_solution = None
        best_profit = float("-inf")

        for restart in range(self.params.restarts):
            # Initialize solution
            routes = build_greedy_routes(
                dist_matrix=self.d,
                wastes=self.wastes,
                capacity=self.Q,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.rng,
            )
            current = Solution(routes, self.d, self.wastes, self.Q, self.R, self.C)

            restart_best = current.copy()
            restart_best_profit = restart_best.profit

            start_time = time.process_time()
            it = 0
            last_improvement_it = 0

            while it < self.params.max_iterations:
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break
                it += 1

                # Select operator using guidance indicators
                operator = self._select_operator()

                # Apply operator
                op_start_time = time.process_time()
                neighbor = self._apply_operator(current, operator)
                op_elapsed = time.process_time() - op_start_time

                # Acceptance decision
                accepted, improvement = self._accept_solution(current, neighbor)

                # Update indicators and scores
                self._update_indicators(operator, improvement, op_elapsed)

                if accepted:
                    current = neighbor

                    if current.profit > restart_best_profit:
                        restart_best = current.copy()
                        restart_best_profit = current.profit
                        last_improvement_it = it

                # Decay acceptance probability and epsilon
                self.accept_worse_prob *= self.params.acceptance_decay
                self.epsilon = max(self.params.min_epsilon, self.epsilon * self.params.epsilon_decay)

                # Restart if stuck
                if it - last_improvement_it > self.params.restart_threshold:
                    break

                getattr(self, "_viz_record", lambda **k: None)(
                    iteration=it,
                    restart=restart,
                    best_profit=restart_best_profit,
                    current_profit=current.profit,
                    operator=operator,
                    improvement=improvement,
                    accepted=accepted,
                    epsilon=self.epsilon,
                )

            # Update global best
            if restart_best_profit > best_profit:
                best_solution = restart_best
                best_profit = restart_best_profit

        if best_solution is None:
            return [], 0.0, 0.0

        return best_solution.routes, best_solution.profit, best_solution.cost

    def _select_operator(self) -> str:
        """
        Select an operator using epsilon-greedy with weighted guidance indicators.

        Follows Chen et al. (2018):
        Score(op) = w1 * IRI(op) + w2 * TBI(op)
        Selection is done via roulette wheel on these normalized scores.

        Returns:
            Operator name.
        """
        all_operators = self.params.move_operators + self.params.perturbation_operators

        # Epsilon-greedy: explore with probability epsilon
        if self.rng.random() < self.epsilon:
            return self.rng.choice(all_operators)

        # Exploit: select based on guidance indicators (Chen 2018, Eq 3)
        combined_scores = []
        for op in all_operators:
            # Raw scores from indicators
            iri_val = self.iri.get_score(op, self.operator_improvements[op])
            tbi_val = self.tbi.get_score(op, self.operator_times[op])

            # Weighted combination (Eq 3)
            # Probability P_i is proportional to combined score
            score = (self.params.iri_weight * iri_val) + (self.params.tbi_weight * tbi_val)
            combined_scores.append(max(1e-6, score))

        # Normalize to probability distribution
        total = sum(combined_scores)
        probabilities = [s / total for s in combined_scores]

        # Selection
        return self.rng.choices(all_operators, weights=probabilities, k=1)[0]

    def _apply_operator(self, solution: Solution, operator: str) -> Solution:
        """
        Apply a low-level heuristic operator to the solution.

        Args:
            solution: Current solution.
            operator: Operator name.

        Returns:
            Modified solution (neighbor).
        """
        neighbor = solution.copy()

        # Move operators (local search)
        if operator in self.params.move_operators:
            neighbor = self._apply_move_operator(neighbor, operator)

        # Perturbation operators (escape local optima)
        elif operator in self.params.perturbation_operators:
            neighbor = self._apply_perturbation_operator(neighbor, operator)

        # Recalculate objective
        neighbor.evaluate()

        return neighbor

    def _apply_move_operator(self, solution: Solution, operator: str) -> Solution:
        """Apply a move operator (intra-route or inter-route)."""
        if not solution.routes or all(len(r) == 0 for r in solution.routes):
            return solution

        # Simple implementation: try a few random moves
        best = solution
        for _ in range(5):
            candidate = solution.copy()

            if "intra" in operator:
                # Intra-route move
                if len(candidate.routes) > 0:
                    route_idx = self.rng.randint(0, len(candidate.routes) - 1)
                    route = candidate.routes[route_idx]
                    if len(route) >= 2:
                        i, j = self.rng.sample(range(len(route)), 2)
                        if "swap" in operator:
                            route[i], route[j] = route[j], route[i]
                        elif "relocate" in operator:
                            node = route.pop(i)
                            route.insert(j, node)
                        elif "two_opt" in operator:
                            route[i : j + 1] = reversed(route[i : j + 1])

            elif "inter" in operator and len(candidate.routes) >= 2:
                r1_idx, r2_idx = self.rng.sample(range(len(candidate.routes)), 2)
                r1, r2 = candidate.routes[r1_idx], candidate.routes[r2_idx]
                if len(r1) > 0 and len(r2) > 0:
                    i = self.rng.randint(0, len(r1) - 1)
                    j = self.rng.randint(0, len(r2) - 1)
                    if "swap" in operator:
                        r1[i], r2[j] = r2[j], r1[i]
                    elif "relocate" in operator:
                        node = r1.pop(i)
                        r2.insert(j, node)
                    elif "exchange" in operator:
                        r1[i], r2[j] = r2[j], r1[i]

            candidate.evaluate()
            # Only accept if feasible and better
            if candidate.is_feasible() and candidate.profit > best.profit:
                best = candidate

        return best

    def _apply_perturbation_operator(self, solution: Solution, operator: str) -> Solution:
        """Apply a perturbation operator to escape local optima."""
        if not solution.routes:
            return solution

        candidate = solution.copy()

        if "removal" in operator:
            # 1. Ruin (Removal)
            all_nodes = [node for route in candidate.routes for node in route]
            if len(all_nodes) > 0:
                if "string" in operator:
                    candidate.routes, _ = string_removal(
                        candidate.routes,
                        max(1, min(len(all_nodes) // 4, 10)),
                        self.d,
                        rng=self.rng,
                    )
                elif "shaw" in operator:
                    candidate.routes, _ = shaw_removal(
                        candidate.routes,
                        max(1, min(len(all_nodes) // 4, 10)),
                        self.d,
                    )
                else:
                    n_remove = max(1, min(len(all_nodes) // 4, 5))
                    to_remove = self.rng.sample(all_nodes, n_remove)
                    for route in candidate.routes:
                        for node in to_remove:
                            if node in route:
                                route.remove(node)

                # 2. Recreate (Re-insertion)
                # Randomly choose between blink and regret-2
                if self.rng.random() < 0.5:
                    candidate.routes = greedy_insertion_with_blinks(
                        candidate.routes,
                        [],
                        self.d,
                        self.wastes,
                        self.Q,
                        blink_rate=self.params.accept_worse_prob,
                        rng=self.rng,
                        expand_pool=True,
                    )
                else:
                    candidate.routes = regret_2_insertion(
                        candidate.routes,
                        [],
                        self.d,
                        self.wastes,
                        self.Q,
                        R=self.R,
                        mandatory_nodes=self.mandatory_nodes,
                    )

        elif "route" in operator:
            # Remove entire route and re-insert nodes
            if len(candidate.routes) > 1:
                candidate.routes.pop(self.rng.randint(0, len(candidate.routes) - 1))
                # Re-fill
                candidate.routes = greedy_insertion_with_blinks(
                    candidate.routes,
                    [],
                    self.d,
                    self.wastes,
                    self.Q,
                    blink_rate=self.params.accept_worse_prob,
                    rng=self.rng,
                    expand_pool=True,
                )

        candidate.evaluate()
        return candidate

    def _accept_solution(self, current: Solution, neighbor: Solution) -> Tuple[bool, float]:
        """
        Decide whether to accept the neighbor solution.

        Args:
            current: Current solution.
            neighbor: Candidate neighbor solution.

        Returns:
            Tuple of (accepted: bool, improvement: float).
        """
        # Never accept infeasible solutions
        if not neighbor.is_feasible():
            return False, 0.0

        improvement = neighbor.profit - current.profit

        # Always accept improvement
        if improvement > 0:
            return True, improvement

        # Accept equal if enabled
        if improvement == 0 and self.params.accept_equal:
            return True, improvement

        # Accept worse with probability
        if self.rng.random() < self.accept_worse_prob:
            return True, improvement

        return False, improvement

    def _update_indicators(self, operator: str, improvement: float, elapsed_time: float) -> None:
        """
        Update guidance indicators based on operator performance.

        Args:
            operator: Operator name.
            improvement: Improvement in objective value.
            elapsed_time: Time taken by operator.
        """
        self.operator_improvements[operator].append(improvement)
        self.operator_times[operator].append(elapsed_time)

        # Update IRI and TBI
        self.iri.update(operator, improvement)
        self.tbi.update(operator, elapsed_time)

    def _roulette_wheel_selection(self, scores: Dict[str, float]) -> str:
        """
        Roulette wheel selection based on operator scores.

        Args:
            scores: Dictionary mapping operator names to scores.

        Returns:
            Selected operator name.
        """
        # Ensure all scores are non-negative
        min_score = min(scores.values())
        if min_score < 0:
            scores = {op: score - min_score + 0.01 for op, score in scores.items()}

        total = sum(scores.values())
        if total == 0:
            return self.rng.choice(list(scores.keys()))

        r = self.rng.uniform(0, total)
        cumulative = 0.0
        for op, score in scores.items():
            cumulative += score
            if cumulative >= r:
                return op

        return list(scores.keys())[-1]


def run_gihh(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args):
    """
    Main GIHH entry point.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of local node indices that MUST be visited.
        *args: Additional arguments (ignored).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    if len(dist_matrix) <= 1:
        return [], 0.0, 0.0

    if len(dist_matrix) == 2:
        d = wastes.get(1, 0)
        if d <= capacity:
            cost = dist_matrix[0][1] + dist_matrix[1][0]
            profit = d * R
            return [[1]], profit, C * cost
        else:
            return [], 0.0, 0.0

    params = GIHHParams(
        time_limit=values.get("time_limit", 60),
        max_iterations=values.get("max_iterations", 1000),
        iri_weight=values.get("iri_weight", 0.6),
        tbi_weight=values.get("tbi_weight", 0.4),
        learning_rate=values.get("learning_rate", 0.1),
        memory_size=values.get("memory_size", 50),
        epsilon=values.get("epsilon", 0.2),
        epsilon_decay=values.get("epsilon_decay", 0.995),
        min_epsilon=values.get("min_epsilon", 0.01),
        accept_equal=values.get("accept_equal", True),
        accept_worse_prob=values.get("accept_worse_prob", 0.05),
        acceptance_decay=values.get("acceptance_decay", 0.99),
        restarts=values.get("restarts", 1),
        restart_threshold=values.get("restart_threshold", 100),
        seed=values.get("seed"),
    )

    solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes, seed=values.get("seed"))
    return solver.solve()
