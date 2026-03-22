"""
String Removal Operator Module.

This module implements the string removal heuristic, which removes contiguous
sequences (strings) of nodes from routes to preserve local structure while
creating gaps for re-insertion.

Also includes profit-based string removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.string import string_removal
    >>> routes, removed = string_removal(routes, n_remove=5, ...)
    >>> from logic.src.policies.other.operators.destroy.string import string_profit_removal
    >>> routes, removed = string_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def string_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers to induce spatial slack.

    The key insight of SISR is that removing adjacent customers creates a large
    contiguous "hole" in the route, providing maneuverability for reinsertion.
    This also reduces the number of re-insertions needed compared to random removal.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix (used for propagation).
        max_string_len: Maximum length of a string to remove.
        avg_string_len: Average string length (unused, kept for API compatibility).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    removed: List[int] = []
    max_iter = n_remove * 3
    iterations = 0

    if rng is None:
        rng = Random(42)

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1
        available_routes = [(i, r) for i, r in enumerate(routes) if r]
        if not available_routes:
            break

        r_idx, route = rng.choice(available_routes)
        seed_pos = rng.randint(0, len(route) - 1)
        seed_node = route[seed_pos]

        if seed_node in removed:
            continue

        # Geometric distribution for string length
        string_len = 1
        while string_len < max_string_len and rng.random() < (1 - 1 / avg_string_len):
            string_len += 1

        string_len = min(string_len, n_remove - len(removed), len(route))
        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        for pos in range(end - 1, start - 1, -1):
            removed.append(routes[r_idx].pop(pos))

        if len(removed) < n_remove and string_nodes:
            _propagate_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, max_string_len)

    routes = [r for r in routes if r]
    return routes, removed


def _propagate_string_removal(
    routes: List[List[int]],
    removed: List[int],
    dist_matrix: np.ndarray,
    seed_nodes: List[int],
    n_remove: int,
    max_string_len: int,
) -> None:
    """Propagate removal to geographic neighbors."""
    neighbor_candidates = []
    for seed in seed_nodes:
        if seed >= len(dist_matrix):
            continue
        distances = dist_matrix[seed]
        for node_id in range(1, len(distances)):
            if node_id not in removed and node_id not in seed_nodes:
                neighbor_candidates.append((node_id, distances[node_id]))

    neighbor_candidates.sort(key=lambda x: (x[1], x[0]))

    for neighbor, _ in neighbor_candidates[:3]:
        if len(removed) >= n_remove:
            break
        for route in routes:
            if neighbor in route:
                pos = route.index(neighbor)
                string_len = min(2, len(route), n_remove - len(removed))
                start = max(0, pos - string_len // 2)
                end = min(len(route), start + string_len)
                for p in range(end - 1, start - 1, -1):
                    if route[p] not in removed:
                        removed.append(route.pop(p))
                break


def string_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers, biased toward low-profit regions (VRPP).

    Selects a low-profit seed node, removes a string of customers starting from it,
    and propagates removal to neighboring regions with similar low marginal profit.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        max_string_len: Maximum length of a string to remove.
        avg_string_len: Average string length (geometric distribution).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    if rng is None:
        rng = Random(42)

    # 1. Pre-calculate marginal profits
    node_profits: Dict[int, float] = {}
    for route in routes:
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            node_profits[node] = revenue - (detour_cost * C)

    all_nodes = list(node_profits.keys())
    if not all_nodes:
        return routes, []

    # 2. Pick low-profit seeds
    all_nodes.sort(key=lambda n: node_profits[n])
    bottom_quartile = max(1, len(all_nodes) // 4)
    low_profit_nodes = all_nodes[:bottom_quartile]

    removed: List[int] = []
    max_iter = n_remove * 3
    iterations = 0

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1
        # Select seed from low-profit pool
        avail = [n for n in low_profit_nodes if n not in removed]
        if not avail:
            # Fallback to any node
            avail = [n for n in all_nodes if n not in removed]
        if not avail:
            break

        seed_node = rng.choice(avail)
        # Find route/pos
        r_idx, seed_pos = -1, -1
        for i, r in enumerate(routes):
            if seed_node in r:
                r_idx, seed_pos = i, r.index(seed_node)
                break
        if r_idx == -1:
            continue

        route = routes[r_idx]
        string_len = 1
        while string_len < max_string_len and rng.random() < (1 - 1 / avg_string_len):
            string_len += 1
        string_len = min(string_len, n_remove - len(removed), len(route))

        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        for p in range(end - 1, start - 1, -1):
            removed.append(routes[r_idx].pop(p))

        if len(removed) < n_remove and string_nodes:
            _propagate_profit_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, node_profits)

    routes = [r for r in routes if r]
    return routes, removed


def _propagate_profit_string_removal(
    routes: List[List[int]],
    removed: List[int],
    dist_matrix: np.ndarray,
    seed_nodes: List[int],
    n_remove: int,
    node_profits: Dict[int, float],
) -> None:
    """Propagate based on profit similarity and distance."""
    avg_seed_p = sum(node_profits.get(n, 0.0) for n in seed_nodes) / len(seed_nodes)

    candidates = []
    for node_id, p in node_profits.items():
        if node_id in removed or node_id in seed_nodes:
            continue
        # Similarity score: distance + scaled profit difference
        min_d = min(dist_matrix[s][node_id] for s in seed_nodes if s < len(dist_matrix))
        p_diff = abs(p - avg_seed_p)
        score = min_d + p_diff * 0.5
        candidates.append((node_id, score))

    candidates.sort(key=lambda x: x[1])

    for neighbor, _ in candidates[:3]:
        if len(removed) >= n_remove:
            break
        for route in routes:
            if neighbor in route:
                pos = route.index(neighbor)
                string_len = min(2, len(route), n_remove - len(removed))
                start = max(0, pos - string_len // 2)
                end = min(len(route), start + string_len)
                for p in range(end - 1, start - 1, -1):
                    if route[p] not in removed:
                        removed.append(route.pop(p))
                break
