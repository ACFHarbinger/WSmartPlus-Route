"""
Random Removal Operator Module.

This module implements the random removal heuristic, which simply removes
a specified number of nodes chosen uniformly at random.

Also includes profit-based random removal (biased random) for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.random import random_removal
    >>> routes, removed = random_removal(routes, n_remove=5)
    >>> from logic.src.policies.other.operators.destroy.random import random_profit_removal
    >>> routes, removed = random_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def random_removal(
    routes: List[List[int]], n_remove: int, rng: Optional[Random] = None
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes randomly from the solution.

    Selects `n_remove` nodes uniformly at random from the current routes
    and removes them.

    Args:
        routes: The current solution (list of routes).
        n_remove: Number of nodes to remove.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing the
        modified routes (with nodes removed) and a list of removed node IDs.
    """
    removed = []

    # Flatten
    all_nodes = []
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            all_nodes.append((r_idx, n_idx, node))

    if not all_nodes:
        return routes, []

    if n_remove >= len(all_nodes):
        # Remove all nodes
        removed = [n for _, _, n in all_nodes]
        return [[] for _ in routes], removed

    if rng is None:
        rng = Random(42)

    targets = rng.sample(all_nodes, n_remove)

    # Sort targets by r_idx, n_idx desc to pop safely
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node in targets:
        routes[r_idx].pop(n_idx)
        removed.append(node)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed


def random_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    bias_strength: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with biased-random selection favoring low-profit nodes (VRPP).

    Uses marginal profit-weighted sampling where nodes with lower profit contribution
    have higher probability of removal. This is softer than deterministic worst-profit
    removal, introducing diversity while still targeting unprofitable nodes.

    Args:
        routes: The current solution (list of routes).
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        bias_strength: Controls bias toward low-profit nodes (default 3.0).
                       Use 0.0 for uniform random, >0 for profit-biased sampling.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed node IDs.
    """
    if rng is None:
        rng = Random(42)

    # Flatten and calculate marginal profits
    all_nodes = []
    node_profits = []
    for r_idx, route in enumerate(routes):
        for n_idx, node in enumerate(route):
            # Marginal profit formula
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if n_idx == 0 else route[n_idx - 1]
            nex = 0 if n_idx == len(route) - 1 else route[n_idx + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            all_nodes.append((r_idx, n_idx, node, profit))
            node_profits.append(profit)

    if not all_nodes:
        return routes, []

    if n_remove >= len(all_nodes):
        removed = [n for _, _, n, _ in all_nodes]
        return [[] for _ in routes], removed

    # Profit-biased sampling
    min_p, max_p = min(node_profits), max(node_profits)
    p_range = max_p - min_p if max_p != min_p else 1.0

    # Weights: lower profit -> higher weight
    weights = []
    for _, _, _, p in all_nodes:
        norm_low_profit = (max_p - p) / p_range
        weight = (norm_low_profit + 0.1) ** bias_strength
        weights.append(weight)

    # Weighted sampling without replacement
    targets = []
    available_indices = list(range(len(all_nodes)))
    available_weights = weights[:]

    for _ in range(n_remove):
        if not available_indices:
            break
        total_w = sum(available_weights)
        if total_w == 0:
            idx = rng.choice(range(len(available_indices)))
        else:
            probs = [w / total_w for w in available_weights]
            cumsum, rand_val = 0.0, rng.random()
            idx = 0
            for i, p in enumerate(probs):
                cumsum += p
                if rand_val <= cumsum:
                    idx = i
                    break

        sel_idx = available_indices.pop(idx)
        available_weights.pop(idx)
        targets.append(all_nodes[sel_idx])

    # Sort and remove
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)
    removed = []
    for r_idx, n_idx, node, _ in targets:
        routes[r_idx].pop(n_idx)
        removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
