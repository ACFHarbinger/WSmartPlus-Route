"""
Hyper-ACO Runner Module.

This module provides a high-level interface to run the Hyper-Heuristic ACO
solver. It handles parameter parsing, initialization, and execution.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco.runner import run_hyper_heuristic_aco
    >>> result = run_hyper_heuristic_aco(dist_matrix, demands, capacity, ...)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .hyper_aco import HyperHeuristicACO
from .params import HyperACOParams


def run_hyper_heuristic_aco(
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    *args: Any,
) -> Tuple[List[List[int]], float, float]:
    """
    Main entry point for Hyper-Heuristic ACO solver.

    Args:
        dist_matrix: Distance matrix.
        demands: Node demands dictionary.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Configuration dictionary with Hyper-ACO parameters.
        *args: Optional initial routes as the first extra argument.

    Returns:
        Tuple[List[List[int]], float, float]: (routes, profit, cost)
    """
    # Parse parameters
    params = HyperACOParams(
        n_ants=values.get("n_ants", 10),
        alpha=values.get("alpha", 1.0),
        beta=values.get("beta", 2.0),
        rho=values.get("rho", 0.1),
        tau_0=values.get("tau_0", 1.0),
        tau_min=values.get("tau_min", 0.01),
        tau_max=values.get("tau_max", 10.0),
        max_iterations=values.get("max_iterations", 50),
        time_limit=values.get("time_limit", 30.0),
        sequence_length=values.get("sequence_length", 5),
        q0=values.get("q0", 0.9),
        operators=values.get("operators"),  # type: ignore[arg-type]
    )

    # Determine initial routes
    initial_routes: Optional[List[List[int]]] = None
    if args:
        initial_routes = args[0]
    elif "initial_routes" in values:
        initial_routes = values["initial_routes"]
    else:
        # Build a simple greedy construction if none provided
        initial_routes = _build_greedy_solution(list(demands.keys()), dist_matrix, demands, capacity)

    solver = HyperHeuristicACO(dist_matrix, demands, capacity, R, C, params)  # type: ignore[arg-type]
    return solver.solve(initial_routes)


def _build_greedy_solution(
    nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
) -> List[List[int]]:
    """Build initial greedy solution using nearest neighbor."""
    # Filter out depot (0) if it's in nodes, but usually demands keys are the nodes to visit
    unvisited = set(n for n in nodes if n != 0)
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
