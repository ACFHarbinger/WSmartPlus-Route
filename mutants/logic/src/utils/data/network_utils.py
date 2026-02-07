"""
Distance matrix utilities and network analysis functions.
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


def apply_edges(
    dist_matrix: np.ndarray, edge_thresh: float, edge_method: Optional[str]
) -> Tuple[np.ndarray, Optional[Dict[Tuple[int, int], List[int]]], Optional[np.ndarray]]:
    """
    Sparsifies distance matrix by removing long edges and computing shortest paths.
    """

    def _make_path(start, end):
        if next_node[start, end] == -1:
            return []  # No path exists
        path = [start]
        while start != end:
            start = next_node[start, end]
            if start == -1:  # Ensure no infinite loops
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
        n_vertices = len(dist_matrix_edges)
        dist_matrix_edges[adj_matrix == 0] = math.inf
        np.fill_diagonal(dist_matrix_edges, np.zeros((n_vertices, n_vertices)))
        next_node = np.full((n_vertices, n_vertices), -1, dtype=int)
        for i in range(n_vertices):
            for j in range(n_vertices):
                if adj_matrix[i][j]:
                    next_node[i][j] = j
                if i == j:
                    next_node[i][j] = i
        for k in range(n_vertices):
            for i in range(n_vertices):
                for j in range(n_vertices):
                    if dist_matrix_edges[i, k] + dist_matrix_edges[k, j] < dist_matrix_edges[i, j]:
                        dist_matrix_edges[i, j] = dist_matrix_edges[i, k] + dist_matrix_edges[k, j]
                        next_node[i, j] = next_node[i, k]

        shortest_paths = {(i, j): _make_path(i, j) for i in range(n_vertices) for j in range(n_vertices) if i != j}
    else:
        shortest_paths = None
    return dist_matrix_edges, shortest_paths, adj_matrix


def get_paths_between_states(
    n_bins: int, shortest_paths: Optional[Dict[Tuple[int, int], List[int]]] = None
) -> List[List[List[int]]]:
    """
    Constructs a nested list of paths between all pairs of bins.
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
