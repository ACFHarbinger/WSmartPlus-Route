"""
Graph search and combinatorial algorithms.
"""

from typing import List, Tuple

import torch


def find_longest_path(dist_matrix: torch.Tensor, start_vertex: int = 0) -> Tuple[float, List[int]]:
    """
    Find the longest path in a DAG represented by a distance matrix.

    Args:
        dist_matrix: n x n tensor where dist_matrix[i][j] is the weight
                     of edge from node i to node j. Use -inf for no edge.
        start_vertex: Starting vertex index (default: 0).

    Returns:
        tuple: (max_length, path) where max_length is the longest path length
               and path is the list of node indices.
    """
    longest_path = []
    n_vertices = dist_matrix.size(0)
    longest_length = torch.tensor(float("-inf"), device=dist_matrix.device)

    def backtrack(current: int, visited: set, path: List[int], current_length: torch.Tensor):
        """
        Recursive backtracking helper to find longest path.
        """
        nonlocal longest_length, longest_path
        # If all nodes are visited, check if we can return to start_node
        if len(path) == n_vertices:
            return_weight = dist_matrix[current][start_vertex]
            if return_weight != float("-inf"):
                total_length = current_length + return_weight
                if total_length > longest_length:
                    longest_length = total_length
                    longest_path = path[:] + [start_vertex]

            # Record the path even if we cannot return to start, if it is the longest so far
            if current_length > longest_length:
                longest_length = current_length
                longest_path = path[:]
            return

        # Update longest path if current path is longer
        if current_length > longest_length:
            longest_length = current_length
            longest_path = path[:]

        # Explore neighbors
        for next_vertex in range(n_vertices):
            if next_vertex not in visited and dist_matrix[current][next_vertex] != float("-inf"):
                visited.add(next_vertex)
                path.append(next_vertex)
                backtrack(
                    next_vertex,
                    visited,
                    path,
                    current_length + dist_matrix[current][next_vertex],
                )
                visited.remove(next_vertex)
                path.pop()

    # Start backtracking from start_vertex
    backtrack(
        start_vertex,
        {start_vertex},
        [start_vertex],
        torch.tensor(0.0, device=dist_matrix.device),
    )
    return float(longest_length.item()), longest_path
