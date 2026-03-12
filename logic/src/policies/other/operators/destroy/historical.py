"""
Historical Node Removal Operator Module.

Uses historical memory to penalize nodes that are frequently found in
high-cost solutions.  Removes the ``k`` nodes with the worst historical
scores from their routes.

Also includes profit-based historical removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.historical import historical_removal
    >>> history = {1: 12.5, 2: 8.0, 3: 15.3, ...}
    >>> routes, removed = historical_removal(routes, n_remove=5, history=history)
    >>> from logic.src.policies.other.operators.destroy.historical import historical_profit_removal
    >>> routes, removed = historical_profit_removal(routes, n_remove=5, history=history,
    ...                                             dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def historical_removal(
    routes: List[List[int]],
    n_remove: int,
    history: Dict[int, float],
    rng: Optional[Random] = None,
    noise: float = 0.1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with the worst historical cost scores.

    Scores each node by its historical penalty (higher = worse).  A small
    random noise term breaks ties and introduces diversity.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        history: Dictionary mapping node IDs to their historical cost scores
                 (moving averages across iterations).
        rng: Random number generator.
        noise: Noise amplitude (fraction of max score) added for diversity.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random(42)

    # Collect all nodes with their scores
    scored: List[Tuple[float, int, int, int]] = []  # (score, node, r_idx, pos)
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            base_score = history.get(node, 0.0)
            scored.append((base_score, node, r_idx, pos))

    if not scored:
        return routes, []

    # Add noise
    max_score = max(s for s, _, _, _ in scored) if scored else 1.0
    noise_amp = noise * max(max_score, 1e-6)
    scored_noisy = [
        (score + rng.uniform(-noise_amp, noise_amp), node, r_idx, pos) for score, node, r_idx, pos in scored
    ]

    # Sort descending by score (worst nodes first)
    scored_noisy.sort(reverse=True)

    n_remove = min(n_remove, len(scored_noisy))
    to_remove = scored_noisy[:n_remove]

    # Sort by (route_idx, position) descending for safe removal
    to_remove.sort(key=lambda x: (x[2], x[3]), reverse=True)

    removed: List[int] = []
    for _, node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def historical_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    history: Dict[int, float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    alpha: float = 0.5,
    rng: Optional[Random] = None,
    noise: float = 0.1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with worst combined historical score and current profit.

    Combines historical penalty with current profit evaluation to remove
    nodes that have been problematic in the past AND have low current profit.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        history: Dictionary mapping node IDs to their historical cost scores
                 (moving averages across iterations).
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        alpha: Weight for historical score (1-alpha for profit score).
               alpha=1.0 uses only history, alpha=0.0 uses only profit.
        rng: Random number generator.
        noise: Noise amplitude (fraction of max score) added for diversity.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random(42)

    # Collect all nodes with their scores
    scored: List[Tuple[float, int, int, int]] = []  # (combined_score, node, r_idx, pos)

    # Calculate profit for each node
    node_profits = {}
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            cost = dist_matrix[0][node] * C
            profit = revenue - cost
            node_profits[node] = profit

    if not node_profits:
        return routes, []

    # Normalize profit to [0, 1] range (lower profit = higher score)
    min_profit = min(node_profits.values())
    max_profit = max(node_profits.values())
    profit_range = max_profit - min_profit if max_profit != min_profit else 1.0

    # Normalize historical scores to [0, 1] range
    hist_values = [history.get(n, 0.0) for n in node_profits.keys()]
    min_hist = min(hist_values) if hist_values else 0.0
    max_hist = max(hist_values) if hist_values else 1.0
    hist_range = max_hist - min_hist if max_hist != min_hist else 1.0

    # Calculate combined scores
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            # Normalize historical score (higher is worse)
            hist_score = (history.get(node, 0.0) - min_hist) / hist_range

            # Normalize profit score (inverted: lower profit = higher score)
            profit_score = (max_profit - node_profits[node]) / profit_range

            # Combined score (higher = worse, should be removed)
            combined = alpha * hist_score + (1 - alpha) * profit_score
            scored.append((combined, node, r_idx, pos))

    if not scored:
        return routes, []

    # Add noise
    max_score = max(s for s, _, _, _ in scored) if scored else 1.0
    noise_amp = noise * max(max_score, 1e-6)
    scored_noisy = [
        (score + rng.uniform(-noise_amp, noise_amp), node, r_idx, pos) for score, node, r_idx, pos in scored
    ]

    # Sort descending by score (worst nodes first)
    scored_noisy.sort(reverse=True)

    n_remove = min(n_remove, len(scored_noisy))
    to_remove = scored_noisy[:n_remove]

    # Sort by (route_idx, position) descending for safe removal
    to_remove.sort(key=lambda x: (x[2], x[3]), reverse=True)

    removed: List[int] = []
    for _, node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
