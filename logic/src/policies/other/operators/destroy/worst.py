"""
Worst Removal Operator Module.

This module implements the worst removal heuristic, which removes the nodes
that cause the highest increase in the objective function (cost/distance).

Follows Ropke & Pisinger (2005) with randomization parameter p >= 1.

Also includes profit-based worst removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.worst import worst_removal
    >>> routes, removed = worst_removal(routes, n_remove=5, dist_matrix=d, p=3.0)
    >>> from logic.src.policies.other.operators.destroy.worst import worst_profit_removal
    >>> routes, removed = worst_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0, p=3.0)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def worst_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes that contribute most to the current routing cost with randomization.

    Implements Ropke & Pisinger (2005) randomized worst removal:
    - Sort nodes by cost savings (detour cost) in descending order
    - For each removal, select node at index: floor(y^p * |L|)
      where y ~ U(0,1) is a random number and L is the sorted candidate list
    - Parameter p >= 1 controls randomness:
      * p = 1: uniform random selection
      * p > 1: biases toward worst nodes (higher p = more deterministic)

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        p: Randomness parameter (p >= 1). Default 1.0 is fully deterministic.
        rng: Random number generator. If None, uses deterministic selection.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    removed = []
    for _ in range(n_remove):
        # Recalculate costs for remaining nodes
        costs = []
        for r_idx, route in enumerate(routes):
            if len(route) == 0:
                continue

            for i, node in enumerate(route):
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) - 1 else route[i + 1]

                # Detour cost (savings from removal)
                savings = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]
                costs.append((r_idx, i, node, savings))

        if not costs:
            break

        # Sort by savings (highest first)
        costs.sort(key=lambda x: x[3], reverse=True)

        # Randomized selection: index = floor(y^p * |L|)
        L = len(costs)
        y = rng.random()  # U(0,1)
        idx = min(int(y**p * L), L - 1)

        r_idx, n_idx, node, _ = costs[idx]

        # Remove the selected node
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx] = [n for n in routes[r_idx] if n != node]
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
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with the worst marginal profit contribution with randomization (VRPP).

    Implements Ropke & Pisinger (2005) randomized worst removal for profit maximization:
    - Calculates Profit_i = (waste_i * R) - (detour_cost_i * C) for each node i
    - Sorts nodes by profit (lowest first, i.e., worst nodes)
    - For each removal, select node at index: floor(y^p * |L|)

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        p: Randomness parameter (p >= 1).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    removed = []
    for _ in range(n_remove):
        # Recalculate profits for remaining nodes
        profits = []
        for r_idx, route in enumerate(routes):
            if len(route) == 0:
                continue

            for i, node in enumerate(route):
                revenue = wastes.get(node, 0.0) * R
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) - 1 else route[i + 1]

                detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
                profit = revenue - (detour_cost * C)
                profits.append((r_idx, i, node, profit))

        if not profits:
            break

        # Sort by profit (lowest first = worst nodes)
        profits.sort(key=lambda x: x[3])

        # Randomized selection
        L = len(profits)
        y = rng.random()
        idx = min(int(y**p * L), L - 1)

        r_idx, n_idx, node, _ = profits[idx]

        # Remove the selected node
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx] = [n for n in routes[r_idx] if n != node]
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
