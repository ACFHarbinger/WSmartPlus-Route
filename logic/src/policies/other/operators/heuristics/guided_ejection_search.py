"""
Guided Ejection Search (GES) Operator Module.

This module implements the Guided Ejection Search heuristic, which ejects
an entire route and reinserts its nodes greedily into the remaining routes.
"""

import random
from typing import Dict, List

import numpy as np

from ..repair import greedy_insertion


def apply_ges(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    rng: random.Random,
) -> List[List[int]]:
    """
    Guided Ejection Search (GES) implementation.

    Acts as a route-level perturbation: selects a random route, ejects all
    its nodes, and forces reinsertion of those nodes into the remaining routes
    minimizing insertion cost.

    Args:
        routes: Current routes.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        rng: Random number generator.

    Returns:
        List[List[int]]: Modified routes.
    """
    if len(routes) <= 1:
        return routes

    new_routes = [r[:] for r in routes if r]
    if not new_routes:
        return new_routes

    # Eject a random route
    ejected_idx = rng.randrange(len(new_routes))
    ejected_nodes = new_routes.pop(ejected_idx)

    # Greedily reinsert into remaining routes
    new_routes = greedy_insertion(
        routes=new_routes,
        removed_nodes=ejected_nodes,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
    )

    return new_routes
