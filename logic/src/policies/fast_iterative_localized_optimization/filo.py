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
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed FILO parameters.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.rng = np.random.default_rng(params.seed)

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Calculate mean arc cost
        total_arcs = max(1, self.n_nodes * (self.n_nodes - 1))
        self.mean_arc_cost = float(np.sum(self.dist_matrix)) / total_arcs

        # Setup R&R
        self.rr = RuinAndRecreate(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            rng=self.rng,
        )

        # Base structures setup
        omega_base = max(1, int(math.ceil(math.log(self.n_nodes + 1))))
        self.omega = [omega_base] * (self.n_nodes + 1)
        self.gamma = [self.params.gamma_base] * (self.n_nodes + 1)
        self.gamma_counter = [0] * (self.n_nodes + 1)

        self.sa_start_temp = self.mean_arc_cost / max(1.0, self.params.initial_temperature_factor)
        self.sa_final_temp = self.sa_start_temp / max(1.0, self.params.final_temperature_factor)

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def build_initial_solution(self) -> List[List[int]]:
        """Greedy constructive heuristic."""
        nodes = self.nodes[:]
        self.rng.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        mandatory_set = set(self.mandatory_nodes) if self.mandatory_nodes else set()

        for node in nodes:
            waste = self.wastes.get(node, 0.0)
            revenue = waste * self.R
            is_mandatory = node in mandatory_set

            if not is_mandatory and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue

            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            routes.append(curr_route)
        return routes

    def apply_local_search(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply a fast local search loop using standard operators."""
        # For simplicity in python we wrap the generic LS manager.
        # A fully optimized version would pass gamma into the C++ style RVND logic.
        ls_manager = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=self.params,
            seed=self.params.seed,
        )
        return ls_manager.optimize(routes)

    def _update_gamma(self, is_new_best: bool, max_non_improving: int) -> None:
        """Update localized gamma parameters based on improvement."""
        for node in range(1, self.n_nodes + 1):
            if is_new_best:
                self.gamma[node] = self.params.gamma_base
                self.gamma_counter[node] = 0
            else:
                self.gamma_counter[node] += 1
                if self.gamma_counter[node] >= max_non_improving:
                    self.gamma[node] = min(self.gamma[node] * 2.0, 1.0)
                    self.gamma_counter[node] = 0

    def _update_omega(
        self,
        ruined: List[int],
        walk_seed: int,
        ls_cost: float,
        current_cost: float,
        shaking_lb: float,
        shaking_ub: float,
    ) -> None:
        """Update dynamic shaking parameters (omega)."""
        seed_shake_val = self.omega[walk_seed]
        if ls_cost > current_cost + shaking_ub:
            for customer in ruined:
                if self.omega[customer] > seed_shake_val - 1:
                    self.omega[customer] = max(1, self.omega[customer] - 1)
        elif current_cost <= ls_cost < current_cost + shaking_lb:
            for customer in ruined:
                if self.omega[customer] < seed_shake_val + 1:
                    self.omega[customer] += 1
        else:
            for customer in ruined:
                if self.rng.random() < 0.5:
                    if self.omega[customer] > seed_shake_val - 1:
                        self.omega[customer] = max(1, self.omega[customer] - 1)
                else:
                    if self.omega[customer] < seed_shake_val + 1:
                        self.omega[customer] += 1

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """Run the FILO algorithm main inner loop."""

        current_routes = copy.deepcopy(initial_solution) if initial_solution else self.build_initial_solution()
        best_routes = copy.deepcopy(current_routes)

        current_cost = self.calculate_cost(current_routes)
        best_cost = current_cost

        overall_customers = sum(len(r) for r in current_routes)
        avg_route_cost = current_cost / max(1.0, (overall_customers + 2.0 * len(current_routes)))

        shaking_lb = avg_route_cost * self.params.shaking_lb_factor
        shaking_ub = avg_route_cost * self.params.shaking_ub_factor

        temperature = self.sa_start_temp
        start_time = time.process_time()

        # Max non-improving iterations depends on the expected length of search
        expected_iterations = self.params.max_iterations
        max_non_improving = math.ceil(self.params.delta_gamma * expected_iterations)

        for iteration in range(self.params.max_iterations):
            elapsed = time.process_time() - start_time
            if self.params.time_limit > 0 and elapsed > self.params.time_limit:
                break

            shaken_routes, walk_seed, ruined = self.rr.apply(current_routes, self.omega, self.nodes)

            # Sub-sample Local search based on gamma to proxy sparsification
            # Note: actual LS Manager loops everywhere, but we trigger the optimization
            ls_routes = self.apply_local_search(shaken_routes)
            ls_cost = self.calculate_cost(ls_routes)

            delta_cost = ls_cost - current_cost

            # Acceptance criteria (minimizing cost)
            accept = False
            if delta_cost < -1e-6:
                accept = True
            elif temperature > 0:
                prob = math.exp(-delta_cost / temperature)
                if self.rng.random() < prob:
                    accept = True

            is_new_best = False
            if ls_cost < best_cost - 1e-6:
                best_routes = copy.deepcopy(ls_routes)
                best_cost = ls_cost
                is_new_best = True

            self._update_gamma(is_new_best, max_non_improving)
            self._update_omega(ruined, walk_seed, ls_cost, current_cost, shaking_lb, shaking_ub)

            if accept:
                current_routes = ls_routes
                current_cost = ls_cost

                # Recalibrate dynamic bounds
                overall_customers = sum(len(r) for r in current_routes)
                avg_route_cost = current_cost / max(1.0, (overall_customers + 2.0 * len(current_routes)))
                shaking_lb = avg_route_cost * self.params.shaking_lb_factor
                shaking_ub = avg_route_cost * self.params.shaking_ub_factor

            # Annealing Schedule
            if temperature > self.sa_final_temp:
                cooling_factor = (self.sa_final_temp / self.sa_start_temp) ** (1.0 / self.params.max_iterations)
                temperature *= cooling_factor

            # Profit formulation
            best_revenue = sum(self.wastes.get(n, 0.0) * self.R for r in best_routes for n in r)
            best_profit = best_revenue - best_cost

            # Visualization
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                current_profit=sum(self.wastes.get(n, 0.0) * self.R for r in current_routes for n in r) - current_cost,
                temperature=temperature,
                accepted=int(accept),
                score=3 if ls_cost < best_cost else (1 if accept else 0),
            )

        best_revenue = sum(self.wastes.get(n, 0.0) * self.R for r in best_routes for n in r)
        best_profit = best_revenue - best_cost

        return best_routes, best_profit, best_cost
