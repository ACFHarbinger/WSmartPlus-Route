"""
Worst Removal Operator Module.

This module implements the worst removal heuristic, which removes the nodes
that cause the highest increase in the objective function (cost/distance).

Also includes profit-based worst removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.worst import worst_removal
    >>> routes, removed = worst_removal(routes, n_remove=5, dist_matrix=d)
    >>> from logic.src.policies.other.operators.destroy.worst import worst_profit_removal
    >>> routes, removed = worst_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from typing import Dict, List, Tuple

import numpy as np


def worst_removal(routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes that contribute most to the current routing cost.

    Calculates the 'cost saving' of removing each node (detour cost) and
    removes those with the highest savings (greedy approach).

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.

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

    # Highest savings first, then tie-break by node ID
    costs.sort(key=lambda x: (x[3], x[2]), reverse=True)
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


def worst_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with the worst marginal profit contribution (VRPP).

    Calculates Profit_i = (waste_i * R) - (detour_cost_i * C) for each node i,
    where detour_cost_i = dist(p-1, i) + dist(i, p+1) - dist(p-1, p+1).
    Removes nodes with the lowest profit values.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Calculate profit for each visited node
    profits = []
    for r_idx, route in enumerate(routes):
        if len(route) == 0:
            continue

        for i, node in enumerate(route):
            # Marginal profit: Revenue - Marginal cost
            revenue = wastes.get(node, 0.0) * R

            # Detour cost (marginal cost) calculation
            # Use 0 (depot) if node is at start/end
            prev = 0 if i == 0 else route[i - 1]
            nex = 0 if i == len(route) - 1 else route[i + 1]

            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            profits.append((r_idx, i, node, profit))

    # Lowest profit first (worst nodes), break ties with node ID
    profits.sort(key=lambda x: (x[3], x[2]))

    # Select candidates for removal
    targets = profits[:n_remove]
    # Sort by index descending to pop safely
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    removed = []
    for r_idx, n_idx, node, _ in targets:
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx].pop(n_idx)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
