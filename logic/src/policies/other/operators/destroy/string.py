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
    max_iter = n_remove * 3  # Prevent infinite loops
    iterations = 0

    if rng is None:
        rng = Random(42)

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1

        # Pick a random seed from remaining nodes
        available_routes = [(i, r) for i, r in enumerate(routes) if r]
        if not available_routes:
            break

        r_idx, route = rng.choice(available_routes)
        if not route:
            continue

        # Pick seed position
        seed_pos = rng.randint(0, len(route) - 1)
        seed_node = route[seed_pos]

        if seed_node in removed:
            continue

        # Determine string length (geometric-like distribution)
        # L ~ 1 + geometric(1/avg_string_len)
        string_len = 1
        while string_len < max_string_len and rng.random() < (1 - 1 / avg_string_len):
            string_len += 1

        # Don't remove more than needed
        remaining = n_remove - len(removed)
        string_len = min(string_len, remaining, len(route))

        # Extract string starting at seed_pos
        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        # Remove string from route (reverse order to maintain indices)
        for pos in range(end - 1, start - 1, -1):
            node = routes[r_idx].pop(pos)
            removed.append(node)

        # Propagate to neighbors: remove strings from adjacent routes
        if len(removed) < n_remove and string_nodes:
            _propagate_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, max_string_len)

    # Clean empty routes
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
    """
    Propagate string removal to neighboring routes.

    After removing a string, look at the spatial neighbors of the removed nodes
    and remove strings from their routes as well. This creates a concentrated
    "disaster zone" across multiple routes.
    """
    # Find neighbors of removed string
    neighbor_candidates = []
    for seed in seed_nodes:
        if seed >= len(dist_matrix):
            continue
        distances = dist_matrix[seed]
        for node_id in range(1, len(distances)):
            if node_id not in removed and node_id not in seed_nodes:
                neighbor_candidates.append((node_id, distances[node_id]))

    # Sort by distance, then node ID for deterministic tie-breaking
    neighbor_candidates.sort(key=lambda x: (x[1], x[0]))

    # Take closest neighbors
    for neighbor, _ in neighbor_candidates[:3]:
        if len(removed) >= n_remove:
            break

        # Find which route contains this neighbor
        for _r_idx, route in enumerate(routes):
            if neighbor in route:
                pos = route.index(neighbor)
                # Remove a small string around this neighbor
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
    Remove contiguous strings of customers, biased toward low-profit nodes.

    Similar to standard string removal, but selects seed nodes from low-profit
    regions. This creates contiguous holes in unprofitable areas of the solution.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        max_string_len: Maximum length of a string to remove.
        avg_string_len: Average string length for geometric distribution.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    if rng is None:
        rng = Random(42)

    # Calculate profit for all nodes
    node_profits: Dict[int, float] = {}
    all_nodes: List[int] = []
    for route in routes:
        for node in route:
            if node not in node_profits:
                revenue = wastes.get(node, 0.0) * R
                cost = dist_matrix[0][node] * C
                profit = revenue - cost
                node_profits[node] = profit
                all_nodes.append(node)

    if not all_nodes:
        return routes, []

    # Sort nodes by profit (ascending - worst first)
    all_nodes.sort(key=lambda n: node_profits[n])

    # Select seed from bottom 25% of profit nodes
    bottom_quartile = max(1, len(all_nodes) // 4)
    low_profit_nodes = all_nodes[:bottom_quartile]

    removed: List[int] = []
    max_iter = n_remove * 3  # Prevent infinite loops
    iterations = 0

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1

        # Pick seed from low-profit nodes still in routes
        available_seeds = [n for n in low_profit_nodes if any(n in r for r in routes) and n not in removed]
        if not available_seeds:
            # Fallback: pick any node
            available_routes = [(i, r) for i, r in enumerate(routes) if r]
            if not available_routes:
                break
            r_idx, route = rng.choice(available_routes)
            if not route:
                continue
            seed_pos = rng.randint(0, len(route) - 1)
            seed_node = route[seed_pos]
        else:
            seed_node = rng.choice(available_seeds)
            # Find seed in routes
            r_idx = -1
            seed_pos = -1
            for i, route in enumerate(routes):
                if seed_node in route:
                    r_idx = i
                    seed_pos = route.index(seed_node)
                    break
            if r_idx == -1:
                continue

        if seed_node in removed:
            continue

        route = routes[r_idx]

        # Determine string length
        string_len = 1
        while string_len < max_string_len and rng.random() < (1 - 1 / avg_string_len):
            string_len += 1

        # Don't remove more than needed
        remaining = n_remove - len(removed)
        string_len = min(string_len, remaining, len(route))

        # Extract string starting at seed_pos
        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        # Remove string from route (reverse order to maintain indices)
        for pos in range(end - 1, start - 1, -1):
            node = routes[r_idx].pop(pos)
            removed.append(node)

        # Propagate to neighbors: remove strings from adjacent routes
        if len(removed) < n_remove and string_nodes:
            _propagate_profit_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, node_profits)

    # Clean empty routes
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
    """
    Propagate string removal to neighboring routes based on profit similarity.

    After removing a string, look at nodes with similar low profit and remove
    strings from their routes as well. This creates a concentrated "disaster zone"
    in low-profit regions.
    """
    # Calculate average profit of seed string
    avg_seed_profit = sum(node_profits.get(n, 0.0) for n in seed_nodes) / max(len(seed_nodes), 1)

    # Find neighbors with similar low profit
    neighbor_candidates = []
    for node_id in node_profits:
        if node_id not in removed and node_id not in seed_nodes:
            # Combine distance and profit similarity
            min_dist = min(dist_matrix[seed][node_id] for seed in seed_nodes if seed < len(dist_matrix))
            profit_diff = abs(node_profits[node_id] - avg_seed_profit)
            # Combined score: favor close nodes with similar low profit
            score = min_dist + profit_diff * 0.5
            neighbor_candidates.append((node_id, score))

    # Sort by combined score (distance + profit similarity)
    neighbor_candidates.sort(key=lambda x: x[1])

    # Take best candidates
    for neighbor, _ in neighbor_candidates[:3]:
        if len(removed) >= n_remove:
            break

        # Find which route contains this neighbor
        for _r_idx, route in enumerate(routes):
            if neighbor in route:
                pos = route.index(neighbor)
                # Remove a small string around this neighbor
                string_len = min(2, len(route), n_remove - len(removed))
                start = max(0, pos - string_len // 2)
                end = min(len(route), start + string_len)

                for p in range(end - 1, start - 1, -1):
                    if route[p] not in removed:
                        removed.append(route.pop(p))
                break
