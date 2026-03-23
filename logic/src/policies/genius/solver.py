"""
GENIUS (GENI + US) Meta-Heuristic for VRPP.

Implements the GENIUS algorithm combining:
- GENI (Generalized Insertion): Construction heuristic with Type I and Type II moves
- US (Unstringing and Stringing): Post-optimization procedure

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..other.operators.repair.geni import geni_insertion, geni_profit_insertion
from ..other.operators.unstringing_stringing.stringing_wrapper import stringing_insertion, stringing_profit_insertion
from ..other.operators.unstringing_stringing.unstringing_wrapper import (
    unstringing_profit_removal,
    unstringing_removal,
)
from .params import GENIUSParams


class GENIUSSolver:
    """
    GENIUS meta-heuristic solver for VRPP.

    Combines GENI insertion with Unstringing/Stringing post-optimization.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GENIUSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the GENIUS solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot.
            wastes: Dictionary mapping node IDs to waste amounts (demands).
            capacity: Vehicle capacity constraint.
            R: Revenue multiplier for profit calculation.
            C: Cost multiplier for profit calculation.
            params: GENIUS algorithm parameters.
            mandatory_nodes: List of nodes that must be visited.
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random(42)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the GENIUS algorithm.

        Returns:
            Tuple of (routes, profit, cost).
                routes: List of routes, each route is a list of node IDs.
                profit: Total profit (revenue - cost).
                cost: Total routing cost.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Step 1: Initial solution using GENI insertion
        routes = self._build_initial_solution_geni()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Step 2: Iterative improvement using US (Unstringing-Stringing)
        for iteration in range(self.params.n_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Apply US cycles for post-optimization
            for _cycle in range(self.params.us_cycles):
                if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                    break

                # Unstringing phase: Remove nodes
                routes_copy = copy.deepcopy(routes)
                n_remove = min(3, sum(len(r) for r in routes_copy))
                if n_remove == 0:
                    break

                try:
                    if self.params.profit_aware_operators:
                        routes_us, removed = unstringing_profit_removal(
                            routes=routes_copy,
                            n_remove=n_remove,
                            unstring_type=self.params.unstring_type,
                            dist_matrix=self.dist_matrix,
                            wastes=self.wastes,
                            R=self.R,
                            C=self.C,
                            rng=self.random,
                        )
                    else:
                        routes_us, removed = unstringing_removal(
                            routes=routes_copy,
                            n_remove=n_remove,
                            unstring_type=self.params.unstring_type,
                            dist_matrix=self.dist_matrix,
                            rng=self.random,
                        )

                    # Stringing phase: Reinsert removed nodes
                    if removed:
                        if self.params.profit_aware_operators:
                            routes_us = stringing_profit_insertion(
                                routes=routes_us,
                                removed_nodes=removed,
                                string_type=self.params.string_type,
                                dist_matrix=self.dist_matrix,
                                wastes=self.wastes,
                                capacity=self.capacity,
                                R=self.R,
                                C=self.C,
                                mandatory_nodes=self.mandatory_nodes,
                                rng=self.random,
                                expand_pool=self.params.vrpp,
                            )
                        else:
                            routes_us = stringing_insertion(
                                routes=routes_us,
                                removed_nodes=removed,
                                string_type=self.params.string_type,
                                dist_matrix=self.dist_matrix,
                                wastes=self.wastes,
                                capacity=self.capacity,
                                mandatory_nodes=self.mandatory_nodes,
                                rng=self.random,
                                expand_pool=self.params.vrpp,
                            )

                        new_profit = self._evaluate(routes_us)

                        # Accept improvement
                        if new_profit > profit:
                            routes = routes_us
                            profit = new_profit

                            if profit > best_profit:
                                best_routes = copy.deepcopy(routes)
                                best_profit = profit

                except Exception:
                    # If US fails, continue with current solution
                    continue

            # Record progress (for visualization/logging hooks)
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Initial solution construction
    # ------------------------------------------------------------------

    def _build_initial_solution_geni(self) -> List[List[int]]:
        """
        Build initial solution using GENI (Generalized Insertion).

        GENI considers Type I and Type II insertions with p-neighborhood restriction.

        Returns:
            List of routes constructed using GENI.
        """
        # Start with empty routes
        routes: List[List[int]] = []

        # All nodes start as unassigned
        all_nodes = set(self.nodes)

        # Use GENI insertion to build the solution
        if self.params.profit_aware_operators:
            routes = geni_profit_insertion(
                routes=routes,
                removed_nodes=list(all_nodes),
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                neighborhood_size=self.params.neighborhood_size,
                rng=self.random,
                expand_pool=self.params.vrpp,
            )
        else:
            routes = geni_insertion(
                routes=routes,
                removed_nodes=list(all_nodes),
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                neighborhood_size=self.params.neighborhood_size,
                rng=self.random,
                expand_pool=self.params.vrpp,
            )

        # Clean up empty routes
        routes = [r for r in routes if r]

        return routes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Calculate profit of a solution.

        Profit = Revenue - Cost
        Revenue = sum of wastes collected * R
        Cost = total distance * C

        Args:
            routes: List of routes.

        Returns:
            Total profit.
        """
        revenue = sum(self.wastes.get(node, 0) for route in routes for node in route) * self.R
        cost = self._cost(routes)
        return revenue - cost

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing cost.

        Args:
            routes: List of routes.

        Returns:
            Total cost (distance * C).
        """
        total_dist = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first node
            total_dist += self.dist_matrix[0, route[0]]
            # Consecutive nodes
            for i in range(len(route) - 1):
                total_dist += self.dist_matrix[route[i], route[i + 1]]
            # Last node to depot
            total_dist += self.dist_matrix[route[-1], 0]
        return total_dist * self.C
