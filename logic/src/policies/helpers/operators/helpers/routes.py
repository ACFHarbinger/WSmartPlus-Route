"""
Pruning Module for Local Search and Large Neighborhood Search Operators.

This module implements the `prune_unprofitable_routes` function, a utility used
by VRPP local search and large neighborhood search operators to filter out routes
that result in a net economic loss. This is crucial for VRPP variants where routes
must generate sufficient revenue to cover their operational costs.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.helpers.routes import prune_unprofitable_routes
    >>> valid_routes = prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
"""

from typing import Dict, List, Set

import numpy as np


def prune_unprofitable_routes(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    mandatory_nodes_set: Set[int],
) -> List[List[int]]:
    """
    Evaluates all routes and removes those that result in a net economic loss,
    unless they contain mandatory nodes that must be served.

    Args:
        routes: List of completed routes after the insertion phase.
        dist_matrix: Distance matrix.
        wastes: Dictionary mapping node ID to waste volume (demand).
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes_set: Set of node IDs that must be serviced.

    Returns:
        List[List[int]]: A filtered list of economically viable routes.
    """
    valid_routes = []

    for route in routes:
        if not route:
            continue

        # 1. Mandatory routes are always kept
        if any(node in mandatory_nodes_set for node in route):
            valid_routes.append(route)
            continue

        # 2. Calculate full route detour cost
        cost = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i], route[i + 1]]
        cost += dist_matrix[route[-1], 0]

        # 3. Calculate total revenue
        revenue = sum(wastes.get(node, 0.0) for node in route) * R

        # 4. Keep if profitable (or effectively break-even to floating point precision)
        profit = revenue - (cost * C)
        if profit >= -1e-4:
            valid_routes.append(route)

    return valid_routes
