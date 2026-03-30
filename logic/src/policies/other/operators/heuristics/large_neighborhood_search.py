"""
Large Neighbourhood Search (LNS) Operator Module.

This module implements the Large Neighbourhood Search heuristic as defined in
Chen et al. (2018), which randomly removes a subset of nodes and reinserts
them greedily.
"""

import copy
import math
import random
from typing import Dict, List

import numpy as np

from ..destroy import random_removal
from ..repair import greedy_insertion


def apply_lns(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
) -> List[List[int]]:
    """
    Large Neighbourhood Search (LNS) implementation.

    Removes a subset of nodes `q = min(0.05 * n, 10)` randomly and reinserts
    them greedily to minimize distance increase.

    Args:
        routes: Current routes.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier (unused but kept for signature consistency).
        C: Cost multiplier (unused but kept for signature consistency).
        rng: Random number generator.

    Returns:
        List[List[int]]: Modified routes.
    """
    total_nodes = sum(len(r) for r in routes)
    if total_nodes == 0:
        return routes

    # Equation from paper: q = min(0.05 * n, 10)
    q = max(1, min(math.ceil(0.05 * total_nodes), 10))

    # 1. Destroy: remove q nodes
    new_routes, removed_nodes = random_removal(copy.deepcopy(routes), q, rng)

    # 2. Repair: Greedy insertion limiting distance
    new_routes = greedy_insertion(
        routes=new_routes,
        removed_nodes=removed_nodes,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
    )

    return new_routes
