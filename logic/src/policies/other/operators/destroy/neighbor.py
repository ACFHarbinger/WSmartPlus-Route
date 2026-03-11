"""
Neighbor Removal Operator Module.

Selects a random seed node, then removes it along with its ``k-1`` nearest
geographic neighbors from their respective routes.

Uses the distance matrix for efficient nearest-neighbor lookup via
``numpy.argpartition``.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.neighbor import neighbor_removal
    >>> routes, removed = neighbor_removal(routes, n_remove=5, dist_matrix=dm)
"""

from random import Random
from typing import List, Optional, Tuple

import numpy as np


def neighbor_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a seed node and its nearest geographic neighbors.

    Args:
        routes: Current solution (list of routes).
        n_remove: Total number of nodes to remove (including seed).
        dist_matrix: Distance matrix ``(N+1, N+1)`` including depot at index 0.
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random(42)

    # Flatten all nodes in the solution
    all_nodes: List[int] = []
    node_loc = {}  # node → (route_idx, position)
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            all_nodes.append(node)
            node_loc[node] = (r_idx, pos)

    if not all_nodes:
        return routes, []

    n_remove = min(n_remove, len(all_nodes))

    # Pick seed
    seed = rng.choice(all_nodes)

    # Find k nearest neighbors via distance matrix
    dists = dist_matrix[seed]
    # Only consider nodes actually in routes
    node_arr = np.array(all_nodes)
    node_dists = dists[node_arr]
    if n_remove >= len(node_arr):
        to_remove = list(node_arr)
    else:
        # argpartition gives indices of the k smallest values
        kth_indices = np.argpartition(node_dists, n_remove)[:n_remove]
        to_remove = list(node_arr[kth_indices])

    # Remove nodes (process routes in reverse position order for safe popping)
    removal_plan: List[Tuple[int, int, int]] = []
    for node in to_remove:
        if node in node_loc:
            r_idx, pos = node_loc[node]
            removal_plan.append((r_idx, pos, node))

    removal_plan.sort(key=lambda x: (x[0], x[1]), reverse=True)

    removed: List[int] = []
    for r_idx, pos, node in removal_plan:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
