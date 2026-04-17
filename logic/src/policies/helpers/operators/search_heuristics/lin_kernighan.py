"""
Simplified Lin-Kernighan style improvement for single routes.

Implements the sequential search variant: at each step, selects the
most improving edge swap from a candidate list rather than trying all pairs.
This is faster than exhaustive 2-opt for large routes at the cost of
missing some improvements.
"""

from typing import List

import numpy as np

from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import two_opt_route


def solve_lk(
    route: List[int],
    dist_matrix: np.ndarray,
    n_candidates: int = 5,
    max_iter: int = 100,
) -> List[int]:
    """
    Apply a simplified Lin-Kernighan style search to a single route.

    For each node t1 in the route, considers the `n_candidates` nearest
    unvisited neighbours as the second node t2 of the swap. Accepts the
    best improving swap found in each pass.

    Args:
        route:         Ordered node list, depot excluded.
        dist_matrix:   Full distance matrix (index 0 = depot).
        n_candidates:  Number of nearest neighbours to consider per node.
        max_iter:      Maximum passes.

    Returns:
        Improved route (depot excluded).
    """
    if len(route) < 4:
        return two_opt_route(route, dist_matrix, max_iter)

    route = list(route)
    full = [0] + route + [0]
    n = len(full)

    # Precompute nearest-neighbour lists (node → top-k closest nodes in route)
    all_nodes = list(set(full))
    nn_lists = {}
    for node in all_nodes:
        dists = sorted([(dist_matrix[node, other], other) for other in all_nodes if other != node], key=lambda x: x[0])
        nn_lists[node] = [other for _, other in dists[:n_candidates]]

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            t1 = full[i]
            for t2 in nn_lists.get(t1, []):
                if t2 not in full:
                    continue
                j = full.index(t2)
                if abs(i - j) < 2 or {i, j} == {0, n - 1}:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                delta = (
                    dist_matrix[full[lo - 1], full[hi]]
                    + dist_matrix[full[lo], full[hi + 1] if hi + 1 < n else 0]
                    - dist_matrix[full[lo - 1], full[lo]]
                    - dist_matrix[full[hi], full[hi + 1] if hi + 1 < n else 0]
                )
                if delta < -1e-9:
                    full[lo : hi + 1] = full[lo : hi + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return full[1:-1]
