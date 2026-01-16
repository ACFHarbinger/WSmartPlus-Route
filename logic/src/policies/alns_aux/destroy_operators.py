"""
Destroy operators for the Adaptive Large Neighborhood Search (ALNS).

This module contains various removal heuristics used to discard nodes
from the current routing solution to explore the search space.
"""

import random
from typing import List, Tuple

import numpy as np


def random_removal(routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
    """
    Randomly remove n_remove nodes from the current routes.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    removed = []
    # Flatten
    all_nodes = []
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            all_nodes.append((r_idx, n_idx, node))

    if n_remove >= len(all_nodes):
        return [[]], [n for _, _, n in all_nodes]

    targets = random.sample(all_nodes, n_remove)

    # Sort targets by r_idx, n_idx desc to pop safely
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node in targets:
        routes[r_idx].pop(n_idx)
        removed.append(node)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed


def worst_removal(routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes that contribute most to the current routing cost.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Remove nodes that contribute most to the cost
    costs = []
    for r_idx, route in enumerate(routes):
        if len(route) == 0:
            continue

        for i, node in enumerate(route):
            # Calc cost without this node
            # Prev -> Next
            prev = 0 if i == 0 else route[i - 1]
            nex = 0 if i == len(route) - 1 else route[i + 1]

            # We save dist(prev, node) + dist(node, nex)
            saved = dist_matrix[prev][node] + dist_matrix[node][nex]
            # We add dist(prev, nex)
            added = dist_matrix[prev][nex]

            # Savings = saved - added
            savings = saved - added
            costs.append((r_idx, i, node, savings))

    costs.sort(key=lambda x: x[3], reverse=True)  # Highest savings first
    removed = []

    # One-shot greedy:
    targets = costs[:n_remove]
    # Sort by index desc
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node, _ in targets:
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx].pop(n_idx)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def cluster_removal(
    routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray, nodes: List[int]
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a cluster of nodes based on spatial proximity (Shaw Removal variant).

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        nodes (List[int]): List of all node IDs.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Pick a random node, remove it and its k nearest neighbors
    if not any(routes):
        return routes, []

    # Pick seed
    seed_route_idx = random.randint(0, len(routes) - 1)
    if not routes[seed_route_idx]:
        return random_removal(routes, n_remove)

    seed_node = random.choice(routes[seed_route_idx])

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

    candidates.sort(key=lambda x: x[1])

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
