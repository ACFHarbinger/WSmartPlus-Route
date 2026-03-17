"""
Fast Iterative Localized Optimization (FILO) policy module.

This module provides the main entry point for the FILO metaheuristic,
incorporating Ruin & Recreate shaking and Local Search via Simulated Annealing.

References:
    Accorsi, L., & Vigo, D. "A fast and scalable heuristic for the solution
    of large-scale capacitated vehicle routing problems", 2021.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.fast_iterative_localized_optimization.params import FILOParams
from logic.src.policies.fast_iterative_localized_optimization.ruin_recreate import (
    RuinAndRecreate,
)
from logic.src.policies.other.local_search.local_search_aco import (
    ACOLocalSearch,
)

# Ensure you place your greedy_initialization.py file in this path or adjust accordingly
from logic.src.policies.other.operators.heuristics.greedy_initialization import (
    build_greedy_routes,
)


class FILOSolver:
    """
    Implementation of Fast Iterative Localized Optimization (FILO) for CVRP/VRPP.

    FILO maintains dynamic, localized parameters (gamma and omega) that restrict
    the search space during Local Search, leading to highly scalable iterations.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: FILOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the FILO solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier per kg.
            C: Cost multiplier per km.
            params: Algorithm hyperparameters.
            mandatory_nodes: List of mandatory nodes.
        """
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

        self.mandatory_nodes = mandatory_nodes or []
        self.mandatory_nodes_set = set(self.mandatory_nodes)

        self.n_nodes = len(dist_matrix) - 1
        self.all_customers = [n for n in self.waste.keys() if n != 0]

        # Use fixed numpy generator for reproducibility
        self.random = random.Random(self.params.seed)
        self.rng = np.random.default_rng(self.params.seed)

        # Initialize the Profit-Aware Ruin & Recreate operator
        self.ruin_recreate = RuinAndRecreate(
            dist_matrix=self.d,
            wastes=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            rng=self.rng,
        )

        # Local search (reordering) runs on distances, so ACO is still valid
        self.local_search = ACOLocalSearch(
            dist_matrix=self.d,
            waste=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            params=self.params,
            seed=self.params.seed,
        )

        self.gamma_base = self.params.gamma_base
        self.gamma = [self.gamma_base] * (self.n_nodes + 1)

        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))
        self.omega = [omega_base] * (self.n_nodes + 1)

    def _evaluate_routes(self, routes: List[List[int]]) -> Tuple[float, float]:
        """Evaluate VRPP cost and profit."""
        total_cost = 0.0
        total_revenue = 0.0
        for route in routes:
            if not route:
                continue

            # Distance calculations
            prev = 0
            for node in route:
                total_cost += self.d[prev, node] * self.C
                total_revenue += self.waste.get(node, 0.0) * self.R
                prev = node
            total_cost += self.d[prev, 0] * self.C

        return total_cost, total_revenue - total_cost

    def _get_omega(self, current_routes: List[List[int]]) -> List[int]:
        """Extract spatial neighborhood bound omega."""
        visited = []
        for r in current_routes:
            visited.extend(r)

        if not visited:
            return []

        probs = []
        for n in visited:
            p = self.gamma[n]
            probs.append(p)

        probs_arr = np.array(probs, dtype=np.float64)
        probs_arr /= probs_arr.sum()

        num_omega = min(len(visited), max(1, int(len(visited) * 0.2)))
        omega = self.rng.choice(visited, size=num_omega, p=probs_arr, replace=False).tolist()
        return omega

    def _update_gamma(self, is_new_best: bool, accepted: bool, ruined: List[int]) -> None:
        """Update localized gamma parameters based on improvement and involvement."""
        if is_new_best:
            # Reset ALL gamma when a new global best is found (as per paper)
            for i in range(1, self.n_nodes + 1):
                self.gamma[i] = self.params.gamma_base
            return

        # If not accepted, increase gamma for the ruined (involved) nodes
        if not accepted:
            for i in ruined:
                self.gamma[i] = min(1.0, self.gamma[i] + self.params.delta_gamma)
        else:
            # If accepted but not new best, we might choose to reset gamma for ruined nodes
            # to keep the search localized in the new successful region
            for i in ruined:
                self.gamma[i] = self.params.gamma_base

    def _update_omega(
        self,
        is_new_best: bool,
        accepted: bool,
        ruined: List[int],
    ) -> None:
        """Update dynamic shaking parameters (omega)."""
        omega_base = max(1, int(math.ceil(self.params.omega_base_multiplier * math.log(self.n_nodes + 1))))

        if is_new_best:
            # Reset ALL omega when a new global best is found
            for i in range(1, self.n_nodes + 1):
                self.omega[i] = omega_base
            return

        if not accepted:
            # Increase shaking intensity for ruined nodes that failed to improve
            for i in ruined:
                # Randomly shaked in the range [omega_base, 2*omega_base] as a diversification proxy
                self.omega[i] = min(self.n_nodes // 2, self.omega[i] + self.rng.integers(1, 3))
        else:
            # Reset for successful nodes
            for i in ruined:
                self.omega[i] = omega_base

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute the FILO heuristic.

        Returns:
            Tuple of (Best Routes, Best Profit, Best Cost)
        """
        start_time = time.process_time()

        # Step 1: Constructive Initialization (Profit Aware & Mandatory Respecting)
        current_routes = build_greedy_routes(
            dist_matrix=self.d,
            wastes=self.waste,
            capacity=self.Q,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

        current_cost, current_profit = self._evaluate_routes(current_routes)

        best_routes = copy.deepcopy(current_routes)
        best_profit = current_profit
        best_cost = current_cost

        # Simulated Annealing Setup
        if current_cost > 0:
            self.sa_start_temp = current_cost / self.params.initial_temperature_factor
            self.sa_final_temp = current_cost / self.params.final_temperature_factor
        else:
            self.sa_start_temp = 100.0
            self.sa_final_temp = 1.0

        temperature = self.sa_start_temp

        for iteration in range(self.params.max_iterations):
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break

            omega = self._get_omega(current_routes)

            # --- SHAKING ---
            new_routes, num_ruined, ruined = self.ruin_recreate.apply(
                current_routes, omega, self.all_customers, self.mandatory_nodes
            )

            # --- LOCAL SEARCH ---
            ls_routes = self.local_search.optimize(new_routes)

            # --- EVALUATION ---
            ls_cost, ls_profit = self._evaluate_routes(ls_routes)
            delta_profit = ls_profit - current_profit

            # Simulated Annealing Move Acceptance
            accept = False
            if delta_profit > 1e-6:
                accept = True
            elif temperature > 0:
                # delta_profit is negative here, so math.exp(delta_profit / temperature) is <= 1
                prob = math.exp(delta_profit / temperature)
                if self.rng.random() < prob:
                    accept = True

            is_new_best = False
            if ls_profit > best_profit + 1e-6:
                best_routes = copy.deepcopy(ls_routes)
                best_profit = ls_profit
                best_cost = ls_cost
                is_new_best = True

            self._update_gamma(is_new_best, accept, ruined)
            self._update_omega(is_new_best, accept, ruined)

            if accept:
                current_routes = ls_routes
                current_profit = ls_profit
                current_cost = ls_cost

            # Annealing Schedule
            if temperature > self.sa_final_temp:
                cooling_factor = (self.sa_final_temp / self.sa_start_temp) ** (1.0 / self.params.max_iterations)
                temperature *= cooling_factor

            # Visualization Support
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=temperature,
                accepted=int(accept),
                score=3 if is_new_best else (1 if accept else 0),
            )

        return best_routes, best_profit, best_cost
