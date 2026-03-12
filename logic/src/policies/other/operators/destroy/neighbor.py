"""
Neighbor Removal Operator Module.

Selects a random seed node, then removes it along with its ``k-1`` nearest
geographic neighbors from their respective routes.

Also includes profit-based neighbor removal for VRPP problems.

Uses the distance matrix for efficient nearest-neighbor lookup via
``numpy.argpartition``.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.neighbor import neighbor_removal
    >>> routes, removed = neighbor_removal(routes, n_remove=5, dist_matrix=dm)
    >>> from logic.src.policies.other.operators.destroy.neighbor import neighbor_profit_removal
    >>> routes, removed = neighbor_profit_removal(routes, n_remove=5, dist_matrix=dm, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

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


def neighbor_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a seed node with low profit and its neighbors with similar low profit.

    Selects a low-profit seed node, then removes nodes with the most similar
    (low) profit values, regardless of geographic proximity.

    Args:
        routes: Current solution (list of routes).
        n_remove: Total number of nodes to remove (including seed).
        dist_matrix: Distance matrix ``(N+1, N+1)`` including depot at index 0.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
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

    # Calculate profit for each node
    node_profits = []
    for node in all_nodes:
        revenue = wastes.get(node, 0.0) * R
        cost = dist_matrix[0][node] * C
        profit = revenue - cost
        node_profits.append((node, profit))

    # Sort by profit (ascending - lowest profit first)
    node_profits.sort(key=lambda x: (x[1], x[0]))

    # Pick seed from bottom 25% (low-profit nodes)
    bottom_quartile_size = max(1, len(node_profits) // 4)
    seed_idx = rng.randint(0, bottom_quartile_size - 1)
    seed_node, seed_profit = node_profits[seed_idx]

    # Find nodes with similar profit to seed
    profit_diffs = []
    for node, profit in node_profits:
        if node == seed_node:
            continue
        diff = abs(profit - seed_profit)
        profit_diffs.append((node, diff))

    # Sort by profit similarity (smaller difference = more similar)
    profit_diffs.sort(key=lambda x: (x[1], x[0]))

    # Take seed + (n_remove-1) most similar profit nodes
    to_remove = [seed_node]
    if n_remove > 1:
        similar_nodes = [x[0] for x in profit_diffs[: n_remove - 1]]
        to_remove.extend(similar_nodes)

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
