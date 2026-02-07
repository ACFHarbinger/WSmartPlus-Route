"""
Hyper-ACO Policy Adapter.

Adapts the Hyper-Heuristic ACO solver to the common policy interface.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..base_routing_policy import BaseRoutingPolicy
from ..hyper_aco import HyperACOParams, HyperHeuristicACO
from .factory import PolicyRegistry


@PolicyRegistry.register("hyper_aco")
class HyperACOPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic ACO policy class.

    Uses ACO to construct sequences of local search operators.
    """

    def _get_config_key(self) -> str:
        """Return config key for Hyper-ACO."""
        return "hyper_aco"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """
        Run Hyper-ACO solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        # Build initial greedy solution
        nodes = list(sub_demands.keys())
        initial_routes = self._build_greedy_solution(nodes, sub_dist_matrix, sub_demands, capacity)

        # Create solver parameters
        # Extract from values, using defaults from HyperACOParams dataclass
        aco_kwargs = {
            "n_ants": values.get("n_ants", 10),
            "alpha": values.get("alpha", 1.0),
            "beta": values.get("beta", 2.0),
            "rho": values.get("rho", 0.1),
            "tau_0": values.get("tau_0", 1.0),
            "tau_min": values.get("tau_min", 0.01),
            "tau_max": values.get("tau_max", 10.0),
            "max_iterations": values.get("max_iterations", 50),
            "time_limit": values.get("time_limit", 30.0),
            "sequence_length": values.get("sequence_length", 5),
            "q0": values.get("q0", 0.9),
        }
        if "operators" in values:
            aco_kwargs["operators"] = values["operators"]

        params = HyperACOParams(**aco_kwargs)

        solver = HyperHeuristicACO(
            dist_matrix=sub_dist_matrix,
            demands=sub_demands,
            capacity=capacity,
            C=cost_unit,
            params=params,
        )

        # Optimize
        routes = solver.solve(initial_routes)

        # Calculate cost
        solver_cost = self._calculate_cost(routes, sub_dist_matrix, cost_unit)

        return routes, solver_cost

    def _build_greedy_solution(
        self,
        nodes: List[int],
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
    ) -> List[List[int]]:
        """Build initial greedy solution using nearest neighbor."""
        if not nodes:
            return []

        unvisited = set(nodes)
        routes = []

        while unvisited:
            route = []
            current = 0  # Start from depot
            route_load = 0.0

            while unvisited:
                best_node = None
                best_dist = float("inf")

                for node in unvisited:
                    demand = demands.get(node, 0)
                    if route_load + demand <= capacity:
                        dist = dist_matrix[current, node]
                        if dist < best_dist:
                            best_dist = dist
                            best_node = node

                if best_node is None:
                    break

                route.append(best_node)
                route_load += demands.get(best_node, 0)
                unvisited.remove(best_node)
                current = best_node

            if route:
                routes.append(route)

        return routes

    def _calculate_cost(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        cost_unit: float,
    ) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total += dist_matrix[route[i], route[i + 1]]
            total += dist_matrix[route[-1], 0]
        return total * cost_unit
