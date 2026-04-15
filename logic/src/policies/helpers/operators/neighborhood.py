"""
Utility functions for routing operators.

This module provides helper functions used across various operators including
GENI, US (Unstringing/Stringing), and other heuristics.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from typing import List

import numpy as np


def get_p_neighborhood(
    target_node: int,
    tour_nodes: List[int],
    dist_matrix: np.ndarray,
    p: int,
) -> List[int]:
    """
    Get the p-neighborhood of a target node from a tour.

    Returns the p nodes from tour_nodes that have the smallest distance to
    target_node based on the distance matrix. This implements the p-neighborhood
    restriction described in Gendreau et al. (1992).

    Args:
        target_node: The node for which to find the p-neighborhood.
        tour_nodes: List of nodes currently in the tour (candidate neighbors).
        dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 is depot.
        p: Size of the neighborhood (number of closest nodes to return).

    Returns:
        List of up to p node indices from tour_nodes, sorted by distance to target_node.
        Returns fewer than p nodes if tour_nodes has fewer than p elements.

    Example:
        >>> dist_matrix = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
        >>> tour_nodes = [1, 2, 3]
        >>> get_p_neighborhood(0, tour_nodes, dist_matrix, p=2)
        [1, 2]  # nodes 1 and 2 are closest to depot (node 0)
    """
    if not tour_nodes:
        return []

    if p <= 0:
        return []

    # If p >= number of tour nodes, return all tour nodes
    if p >= len(tour_nodes):
        return list(tour_nodes)

    # Get distances from target_node to all tour_nodes
    tour_nodes_array = np.array(tour_nodes)
    distances = dist_matrix[target_node, tour_nodes_array]

    # Get indices of p smallest distances
    p_indices = np.argsort(distances)[:p]

    # Return the actual node IDs (not array indices)
    return tour_nodes_array[p_indices].tolist()
