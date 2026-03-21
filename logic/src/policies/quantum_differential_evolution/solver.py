"""
Quantum-Inspired Differential Evolution (QDE) for VRPP.

Represents each candidate as a quantum amplitude vector q ∈ [0,1]^N.
Standard DE mutation / crossover operates in this continuous space; the
trial vector is collapsed to a discrete routing solution by ranking node
amplitudes and calling greedy_insertion.

Reference:
    Li, B., & Li, P. (2015). "Quantum Inspired Differential Evolution Algorithm."
    Chen, S., Li, Z., Yang, B., & Rudolph, G. (2015). "Quantum-inspired hyper-heuristics for energy-aware scheduling on heterogeneous computing systems."
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ant_colony_optimization_k_sparse.params import KSACOParams
from .params import QDEParams


class QDESolver:
    """
    Quantum-Inspired Differential Evolution solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: QDEParams,
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
        self.n_nodes = len(dist_matrix) - 1  # Exclude depot
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(seed) if seed is not None else random.Random()
        self.np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(42)

        # Initialize Local Search for post-collapse refinement
        from logic.src.policies.other.local_search.local_search_aco import ACOLocalSearch

        aco_params = KSACOParams(local_search_iterations=self.params.local_search_iterations)
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run QDE and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()
        pop_size = self.params.pop_size

        # Initialise population: Q-bits (alpha, beta) where |alpha|^2 + |beta|^2 = 1
        # We store theta where alpha = cos(theta), beta = sin(theta)
        # Initialization to 1/sqrt(2) means theta = pi/4
        thetas = np.full((pop_size, self.n_nodes), np.pi / 4.0)

        # Collapse initial population
        routes_pop = [self._collapse(t) for t in thetas]
        profits = [self._evaluate(r) for r in routes_pop]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(routes_pop[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        # Track individual bests
        pbest_thetas = copy.deepcopy(thetas)
        pbest_profits = copy.deepcopy(profits)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            for i in range(pop_size):
                # --- Quantum Rotation Gate Update ---
                # Update theta based on best global solution and individual best
                # Standard QEA update rule: theta = theta + delta_theta * s(alpha, beta)
                # Here we use a simplified version common in QDE:
                # delta_theta_i = F * (best_theta - theta_i) + random * (pbest_theta - theta_i)

                # Comparison with global best to determine rotation direction
                for j in range(self.n_nodes):
                    # Measure current best solution at node j
                    # (In continuous QDE, we often use the best theta directly)
                    target_theta = thetas[best_idx][j]

                    # Update rule using rotation gates
                    # Delta theta = delta_theta * sign(A)
                    # where A is usually determined by a lookup table based on
                    # current fitness vs best fitness and bit values.
                    # We implement the standard QDE update:
                    thetas[i][j] += self.params.delta_theta * np.sign(target_theta - thetas[i][j])

                # --- Differential Mutation on Thetas (Hybrid QDE) ---
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = self.random.sample(candidates, 3)
                mutant = thetas[r1] + self.params.F * (thetas[r2] - thetas[r3])

                # --- Crossover ---
                j_rand = self.random.randint(0, self.n_nodes - 1)
                trial_theta = np.where(
                    (self.np_rng.uniform(0.0, 1.0, self.n_nodes) < self.params.CR)
                    | (np.arange(self.n_nodes) == j_rand),
                    mutant,
                    thetas[i],
                )

                # Keep thetas in valid range [0, pi/2] for mapping to [0, 1] probability
                trial_theta = np.clip(trial_theta, 0.0, np.pi / 2.0)

                # --- Collapse → discrete routes ---
                trial_routes = self._collapse(trial_theta)
                trial_profit = self._evaluate(trial_routes)

                # --- Selection ---
                if trial_profit >= profits[i]:
                    thetas[i] = trial_theta
                    routes_pop[i] = trial_routes
                    profits[i] = trial_profit

                    if trial_profit > pbest_profits[i]:
                        pbest_profits[i] = trial_profit
                        pbest_thetas[i] = copy.deepcopy(trial_theta)

                    if trial_profit > best_profit:
                        best_routes = copy.deepcopy(trial_routes)
                        best_profit = trial_profit
                        best_cost = self._cost(best_routes)
                        best_idx = i

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=pop_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collapse(self, thetas: np.ndarray) -> List[List[int]]:
        """
        Collapse quantum state (thetas) to a discrete routing solution.

        Measurement: Each node j is selected with probability |beta_j|^2 = sin^2(theta_j).
        Selected nodes are then ordered by their absolute probability amplitudes
        and passed to greedy_insertion.

        Args:
            thetas: Quantum rotation angles.

        Returns:
            Routes built by greedy_insertion.
        """
        # Measurement phase: probabilistic selection based on |beta|^2
        probabilities = np.sin(thetas) ** 2
        selected_mask = self.np_rng.uniform(0.0, 1.0, self.n_nodes) < probabilities

        # Ranking phase: order nodes by selection probability
        ranked = sorted(
            [j for j in range(self.n_nodes) if selected_mask[j]], key=lambda j: probabilities[j], reverse=True
        )

        selected: List[int] = []
        mandatory_set = set(self.mandatory_nodes)
        total_load = 0.0

        for j in ranked:
            node = j + 1  # 1-based index (depot is 0)
            waste = self.wastes.get(node, 0.0)
            if node in mandatory_set or total_load + waste <= self.capacity:
                selected.append(node)
                total_load += waste

        if not selected:
            return []

        from logic.src.policies.other.operators.repair.greedy import (
            greedy_insertion,
        )

        try:
            routes = greedy_insertion(
                [],
                selected,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=False,
            )
            # Apply local search refinement
            return self.ls.optimize(routes)
        except Exception:
            return []

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Compute net profit for a set of routes.

        Args:
            routes: List of routes (each a list of node indices).

        Returns:
            Net profit (revenue − travel cost).
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        cost = self._cost(routes)
        return rev - cost * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Compute total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance.
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
