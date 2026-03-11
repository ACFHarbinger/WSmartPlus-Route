"""
Historical Node Removal Operator Module.

Uses historical memory to penalize nodes that are frequently found in
high-cost solutions.  Removes the ``k`` nodes with the worst historical
scores from their routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.historical import historical_removal
    >>> history = {1: 12.5, 2: 8.0, 3: 15.3, ...}
    >>> routes, removed = historical_removal(routes, n_remove=5, history=history)
"""

from random import Random
from typing import Dict, List, Optional, Tuple


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
