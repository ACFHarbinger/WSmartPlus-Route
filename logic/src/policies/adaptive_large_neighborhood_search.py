"""
Adaptive Large Neighborhood Search (ALNS) policy module.

This module provides the main entry points for the ALNS metaheuristic,
dispatching to specialized implementations based on configuration.

Reference:
    Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems.
    Computers & Operations Research, 34(8), 2403-2435.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alns_aux.alns_package import run_alns_package
from .alns_aux.destroy_operators import cluster_removal, random_removal, worst_removal
from .alns_aux.ortools_wrapper import run_alns_ortools
from .alns_aux.params import ALNSParams
from .alns_aux.repair_operators import greedy_insertion, regret_2_insertion


class ALNSSolver:
    """
    Custom implementation of Adaptive Large Neighborhood Search for CVRP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
    ):
        """
        Initialize the ALNS solver.

        Args:
            dist_matrix: NxN distance matrix.
            demands: Dictionary of node demands.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed ALNS parameters.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Operator registry
        self.destroy_ops = [
            lambda r, n: random_removal(r, n),
            lambda r, n: worst_removal(r, n, self.dist_matrix),
            lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes),
        ]
        self.repair_ops = [
            lambda r, n: greedy_insertion(r, n, self.dist_matrix, self.demands, self.capacity),
            lambda r, n: regret_2_insertion(r, n, self.dist_matrix, self.demands, self.capacity),
        ]

        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run the ALNS algorithm.

        Args:
            initial_solution: Optional starting solution. If None, a constructive heuristic is used.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        if initial_solution:
            current_routes = initial_solution
        else:
            current_routes = self.build_initial_solution()

        best_routes = copy.deepcopy(current_routes)
        best_cost = self.calculate_cost(best_routes)
        current_cost = best_cost

        T = self.params.start_temp
        start_time = time.time()

        for it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            d_idx = self.select_operator(self.destroy_weights)
            r_idx = self.select_operator(self.repair_weights)

            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]

            n_remove = random.randint(
                self.params.min_removal,
                max(
                    self.params.min_removal,
                    int(self.n_nodes * self.params.max_removal_pct),
                ),
            )

            partial_routes, removed = destroy_op(copy.deepcopy(current_routes), n_remove)
            new_routes = repair_op(partial_routes, removed)
            new_cost = self.calculate_cost(new_routes)

            accept = False
            score = 0

            delta = new_cost - current_cost
            if delta < -1e-6:
                accept = True
                if new_cost < best_cost - 1e-6:
                    best_routes = copy.deepcopy(new_routes)
                    best_cost = new_cost
                    score = 3
                else:
                    score = 1
            else:
                prob = math.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    accept = True

            if accept:
                current_routes = new_routes
                current_cost = new_cost

            # Update weights
            lambda_decay = 0.8
            self.destroy_weights[d_idx] = lambda_decay * self.destroy_weights[d_idx] + (1 - lambda_decay) * max(
                0.1, score
            )
            self.repair_weights[r_idx] = lambda_decay * self.repair_weights[r_idx] + (1 - lambda_decay) * max(
                0.1, score
            )

            T *= self.params.cooling_rate

        collected_revenue = sum(self.demands.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        profit = collected_revenue - best_cost

        return best_routes, profit, best_cost

    def select_operator(self, weights: List[float]) -> int:
        """
        Select an operator index based on their weights using roulette wheel selection.

        Args:
            weights: List of operator weights.

        Returns:
            int: Index of the selected operator.
        """
        total = sum(weights)
        r = random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate the total routing cost for a set of routes.

        Args:
            routes: List of routes.

        Returns:
            float: Total distance * cost multiplier.
        """
        total_dist = 0
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
        """
        Build a basic feasible solution using a greedy constructive heuristic.

        Returns:
            List[List[int]]: Initial routes.
        """
        nodes = self.nodes[:]
        random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        for node in nodes:
            demand = self.demands.get(node, 0)
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


def run_alns(dist_matrix, demands, capacity, R, C, values, *args):
    """
    Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        dist_matrix: Distance matrix.
        demands: Bin demands.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    variant = values.get("variant") or values.get("engine") or "custom"

    if variant == "package":
        return run_alns_package(dist_matrix, demands, capacity, R, C, values)
    elif variant == "ortools":
        return run_alns_ortools(dist_matrix, demands, capacity, R, C, values)

    # Default: Custom internal ALNS solver
    params = ALNSParams(
        time_limit=values.get("time_limit", 10),
        max_iterations=values.get("Iterations", 2000),
    )
    solver = ALNSSolver(dist_matrix, demands, capacity, R, C, params)
    return solver.solve()
