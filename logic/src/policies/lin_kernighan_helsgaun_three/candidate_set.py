from __future__ import annotations

import warnings
from collections import deque
from typing import Dict, List

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# ---------------------------------------------------------------------------
# Alpha-measure and candidate-set construction
#
# NOTE: The α-measure as implemented here ignores π-penalties (subgradient
# optimisation is NOT implemented).  True LKH-3 computes π via iterative
# 1-tree lower-bound ascent; here α(i,j) = c(i,j) − β(i,j) with the raw
# edge costs.
# ---------------------------------------------------------------------------


def _compute_all_pairs_max_edge(mst_adj: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the maximum edge weight on the unique MST path between all pairs.

    Runs a BFS from EACH node to ALL other nodes, tracking the heaviest edge
    on the path.  Since the MST is a tree, BFS visits each node exactly once
    and naturally discovers the unique path.

    Complexity: O(N²) — each BFS is O(N) on a tree, and we run N of them.

    Args:
        mst_adj: (n × n) adjacency matrix of the MST (upper-triangular sparse,
                 as returned by ``scipy.sparse.csgraph.minimum_spanning_tree``
                 converted to dense).
        n: Number of nodes.

    Returns:
        (n × n) symmetric matrix where entry [i, j] is the maximum edge weight
        on the unique MST path from i to j.  Diagonal entries are 0.
    """
    max_edge = np.zeros((n, n), dtype=float)

    # Build explicit adjacency list for faster traversal (avoid scanning
    # the full row each time).
    adj: Dict[int, List[tuple]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            w = mst_adj[i, j]
            if w > 0:
                adj[i].append((j, w))
                adj[j].append((i, w))

    for start in range(n):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        queue: deque = deque()
        queue.append((start, 0.0))

        while queue:
            current, path_max = queue.popleft()
            max_edge[start, current] = path_max

            for neighbor, weight in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    new_max = max(path_max, weight)
                    queue.append((neighbor, new_max))

        # Safety check: warn if any node is unreachable in a supposedly
        # connected graph (i.e., max_edge == 0 for a pair that is not self).
        for end in range(n):
            if end != start and not visited[end]:
                warnings.warn(
                    f"MST is disconnected: no path from node {start} to node {end}. "
                    f"α-measures involving this pair will be artificially high.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return max_edge


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute α-nearness for every edge using minimum-spanning-tree sensitivity.

    α(i, j) = c(i, j) − β(i, j),  where β(i, j) is the weight of the heaviest
    edge on the unique MST path between i and j (Helsgaun 2000, Section 4.1).

    Edges with α = 0 belong to some MST; edges with small α are "nearly
    spanning" and hence strong candidates for an optimal tour.

    NOTE: This implementation does NOT use subgradient-optimised π-penalties.
    The α-measure is computed on raw edge costs only.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        (n × n) array of α-values (symmetric, non-negative).

    Complexity:
        O(N² log N) for MST construction + O(N²) for all-pairs max-edge.
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst = mst_sparse.toarray()

    # Single O(N²) pass instead of O(N³)
    beta = _compute_all_pairs_max_edge(mst, n)

    alpha = distance_matrix - beta
    # Ensure non-negative (numerical precision)
    np.maximum(alpha, 0.0, out=alpha)
    # Zero diagonal
    np.fill_diagonal(alpha, 0.0)

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray,
    alpha_measures: np.ndarray,
    max_candidates: int = 5,
) -> Dict[int, List[int]]:
    """
    Build per-node candidate lists sorted by α-nearness (ties broken by cost).

    Restricting the inner LK search to these lists reduces worst-case
    complexity from O(n²) to O(n · max_candidates) per pass.

    Args:
        distance_matrix: (n × n) cost matrix.
        alpha_measures: (n × n) α-nearness matrix from :func:`compute_alpha_measures`.
        max_candidates: Maximum number of candidates per node (default 5).

    Returns:
        dict mapping each node index to its sorted candidate list.
    """
    n = len(distance_matrix)
    candidates: Dict[int, List[int]] = {}
    for i in range(n):
        indices = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: (alpha_measures[i, j], distance_matrix[i, j]),
        )
        candidates[i] = indices[:max_candidates]
    return candidates
