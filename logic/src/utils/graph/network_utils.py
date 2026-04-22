"""
Distance matrix utilities and network analysis functions.

Attributes:
    _floyd_warshall: Floyd-Warshall algorithm for all-pairs shortest paths.
    apply_edges: Sparsifies distance matrix and computes shortest paths.
    get_paths_between_states: Constructs a nested list of paths between all pairs of bins.

Example:
    _floyd_warshall(dist_matrix, adj_matrix)
    apply_edges(dist_matrix, edge_thresh, edge_method)
    get_paths_between_states(n_bins, shortest_paths)
"""

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.graph import (
    get_adj_knn,
    get_edge_idx_dist,
    idx_to_adj,
)


def _floyd_warshall(dist_matrix: np.ndarray, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.

    Args:
        dist_matrix: Distance matrix.
        adj_matrix: Adjacency matrix.

    Returns:
        Tuple of (distance matrix, next node matrix).
    """
    n_vertices = len(dist_matrix)
    dist_matrix[adj_matrix == 0] = math.inf
    np.fill_diagonal(dist_matrix, 0)
    next_node = np.full((n_vertices, n_vertices), -1, dtype=int)

    for i in range(n_vertices):
        for j in range(n_vertices):
            if adj_matrix[i, j]:
                next_node[i, j] = j
            if i == j:
                next_node[i, j] = i

    for k in range(n_vertices):
        for i in range(n_vertices):
            for j in range(n_vertices):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
                    next_node[i, j] = next_node[i, k]
    return dist_matrix, next_node


def apply_edges(
    dist_matrix: np.ndarray, edge_thresh: float, edge_method: Optional[str]
) -> Tuple[np.ndarray, Optional[Dict[Tuple[int, int], List[int]]], Optional[np.ndarray]]:
    """
    Sparsifies distance matrix and computes shortest paths.

    Args:
        dist_matrix: Distance matrix.
        edge_thresh: Edge threshold.
        edge_method: Edge method.

    Returns:
        Tuple of (distance matrix, shortest paths, adjacency matrix).
    """

    def _make_path(start, end, next_node):
        """
        Make a path from start to end.

        Args:
            start: Start node.
            end: End node.
            next_node: Next node matrix.

        Returns:
            List of nodes in the path.
        """
        if next_node[start, end] == -1:
            return []
        path = [start]
        while start != end:
            start = next_node[start, end]
            if start == -1:
                return []
            path.append(start)
        return path

    dist_matrix_edges = deepcopy(dist_matrix)
    if edge_thresh > 0 and edge_method == "dist":
        adj_matrix = idx_to_adj(get_edge_idx_dist(dist_matrix_edges[1:, 1:], edge_thresh, undirected=False))
    elif edge_thresh > 0 and edge_method == "knn":
        adj_matrix = get_adj_knn(dist_matrix_edges[1:, 1:], edge_thresh, negative=False)
    else:
        adj_matrix = None

    if adj_matrix is not None:
        dist_matrix_edges, next_node = _floyd_warshall(dist_matrix_edges, adj_matrix)
        n_vertices = len(dist_matrix_edges)
        shortest_paths = {
            (i, j): _make_path(i, j, next_node) for i in range(n_vertices) for j in range(n_vertices) if i != j
        }
    else:
        shortest_paths = None
    return dist_matrix_edges, shortest_paths, adj_matrix


def get_paths_between_states(
    n_bins: int, shortest_paths: Optional[Dict[Tuple[int, int], List[int]]] = None
) -> List[List[List[int]]]:
    """
    Constructs a nested list of paths between all pairs of bins.

    Args:
        n_bins: Number of bins.
        shortest_paths: Shortest paths between nodes.

    Returns:
        Nested list of paths between all pairs of bins.
    """
    paths_between_states: List[List[List[int]]] = []
    for ii in range(0, n_bins):
        paths_between_states.append([])
        for jj in range(n_bins):
            if shortest_paths is None or ii == jj:
                paths_between_states[ii].append([ii, jj])
            else:
                paths_between_states[ii].append(shortest_paths[(ii, jj)])
    return paths_between_states
