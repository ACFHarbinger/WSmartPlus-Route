"""
Cluster Removal Operator Module.

This module implements the cluster removal heuristic, which removes a cluster of
nodes based on spatial proximity, effectively acting as a variant of Shaw removal.

Also includes profit-based cluster removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.cluster import cluster_removal
    >>> routes, removed = cluster_removal(routes, n_remove=5, dist_matrix=d, nodes=all_nodes)
    >>> from logic.src.policies.other.operators.destroy.cluster import cluster_profit_removal
    >>> routes, removed = cluster_profit_removal(routes, n_remove=5, dist_matrix=d,
    ...                                          wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .random import random_removal


def cluster_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    nodes: List[int],
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a cluster of nodes based on spatial proximity (Shaw Removal variant).

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        nodes (List[int]): List of all node IDs.
        rng (Optional[Random]): Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Pick a random node, remove it and its k nearest neighbors
    if not any(routes):
        return routes, []

    if rng is None:
        rng = Random(42)

    # Pick seed
    seed_route_idx = rng.randint(0, len(routes) - 1)
    if not routes[seed_route_idx]:
        return random_removal(routes, n_remove, rng=rng)

    seed_node = rng.choice(routes[seed_route_idx])

    removed = [seed_node]

    # Get all nodes current pos
    node_map = {}
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            node_map[node] = (r_idx, n_idx)

    # Find neighbors
    candidates = []
    for v in nodes:
        if v == seed_node or v not in node_map:
            continue
        dist = dist_matrix[seed_node][v]
        candidates.append((v, dist))

    # Sort by distance, then node ID for deterministic tie-breaking
    candidates.sort(key=lambda x: (x[1], x[0]))

    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed.extend(target_nodes)

    # Now remove them from routes
    to_remove_locs = []
    for node in removed:
        if node in node_map:
            to_remove_locs.append((*node_map[node], node))

    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, n_idx, node in to_remove_locs:
        routes[r_idx].pop(n_idx)
        final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed


def cluster_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a cluster of nodes based on low profit potential.

    Selects a seed node with low profit, then removes nodes with similar
    (low) profit values. This targets unprofitable clusters for removal.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        wastes (Dict[int, float]): Waste/profit values for each node.
        R (float): Revenue per unit waste.
        C (float): Cost per unit distance.
        rng (Optional[Random]): Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Pick a node with low profit, remove it and similar low-profit neighbors
    if not any(routes):
        return routes, []

    if rng is None:
        rng = Random(42)

    # Get all nodes currently in routes
    node_map = {}
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            node_map[node] = (r_idx, n_idx)

    if not node_map:
        return routes, []

    # Calculate profit for each node in routes
    # Profit = Revenue - Cost (to reach from depot)
    node_profits = []
    for node in node_map.keys():
        revenue = wastes.get(node, 0.0) * R
        cost = dist_matrix[0][node] * C  # Distance from depot
        profit = revenue - cost
        node_profits.append((node, profit))

    # Sort by profit (ascending - lowest profit first)
    node_profits.sort(key=lambda x: (x[1], x[0]))

    # Pick seed from bottom 25% (low-profit nodes)
    bottom_quartile_size = max(1, len(node_profits) // 4)
    seed_idx = rng.randint(0, bottom_quartile_size - 1)
    seed_node, seed_profit = node_profits[seed_idx]

    removed = [seed_node]

    # Find nodes with similar low profit
    candidates = []
    for node, profit in node_profits:
        if node == seed_node or node not in node_map:
            continue
        # Measure similarity by profit difference
        profit_diff = abs(profit - seed_profit)
        candidates.append((node, profit_diff))

    # Sort by profit similarity (smaller difference = more similar)
    candidates.sort(key=lambda x: (x[1], x[0]))

    # Select n_remove-1 most similar low-profit nodes
    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed.extend(target_nodes)

    # Remove from routes
    to_remove_locs = []
    for node in removed:
        if node in node_map:
            to_remove_locs.append((*node_map[node], node))

    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, n_idx, node in to_remove_locs:
        routes[r_idx].pop(n_idx)
        final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed
