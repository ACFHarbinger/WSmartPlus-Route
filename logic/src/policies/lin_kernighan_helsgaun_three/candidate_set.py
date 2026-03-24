from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# ---------------------------------------------------------------------------
# Alpha-measure and candidate-set construction
# ---------------------------------------------------------------------------


def _find_mst_path_max(mst_adj: np.ndarray, start: int, end: int, n: int) -> float:
    """
    Find maximum edge weight on the unique path in an MST between two nodes.

    Uses BFS to trace the path in the (undirected) MST and returns the weight
    of the heaviest edge encountered.

    Args:
        mst_adj: (n × n) adjacency matrix of the MST (upper-triangular sparse).
        start: Source node index.
        end: Destination node index.
        n: Number of nodes.

    Returns:
        Maximum edge weight on the path; 0.0 if start == end or no path exists.
    """
    if start == end:
        return 0.0

    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    queue = [start]
    visited[start] = True
    while queue:
        current = queue.pop(0)
        if current == end:
            break
        for neighbor in range(n):
            if not visited[neighbor] and (mst_adj[current, neighbor] > 0 or mst_adj[neighbor, current] > 0):
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)

    if parent[end] == -1:
        return 0.0

    max_edge = 0.0
    current = end
    while parent[current] != -1:
        prev = parent[current]
        edge_weight = max(mst_adj[prev, current], mst_adj[current, prev])
        max_edge = max(max_edge, edge_weight)
        current = prev

    return max_edge


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute α-nearness for every edge using minimum-spanning-tree sensitivity.

    α(i, j) = c(i, j) − β(i, j),  where β(i, j) is the weight of the heaviest
    edge on the unique MST path between i and j (Helsgaun 2000, Section 4.1).

    Edges with α = 0 belong to some MST; edges with small α are "nearly
    spanning" and hence strong candidates for an optimal tour.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        (n × n) array of α-values (symmetric, non-negative).
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst = mst_sparse.toarray()

    alpha = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            max_mst_edge = _find_mst_path_max(mst, i, j, n)
            alpha_val = distance_matrix[i, j] - max_mst_edge
            alpha[i, j] = alpha_val
            alpha[j, i] = alpha_val

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
