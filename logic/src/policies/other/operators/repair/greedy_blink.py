"""
Greedy Blink Insertion Operator Module.

This module implements the 'Greedy with Blinks' insertion heuristic based on
the Slack Induction by String Removal (SISR) metaheuristic.

It contains:
    1. `greedy_insertion_with_blinks`: A standard distance-minimizing operator.
    2. `greedy_profit_insertion_with_blinks`: A VRPP-specific profit-maximizing
       operator that utilizes speculative seeding and post-insertion pruning.
    3. `prune_unprofitable_routes`: A safety-net helper for VRPP economics.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.greedy_blink import greedy_insertion_with_blinks
    >>> routes = greedy_insertion_with_blinks(routes, removed, dist_matrix, wastes, capacity, blink_rate=0.1)
"""

from random import Random
from typing import Dict, List, Optional, Set

import numpy as np


def prune_unprofitable_routes(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    mandatory_nodes_set: Set[int],
) -> List[List[int]]:
    """
    Evaluates all routes and removes those that result in a net economic loss,
    unless they contain mandatory nodes that must be served.

    Args:
        routes: List of completed routes after the insertion phase.
        dist_matrix: Distance matrix.
        wastes: Dictionary mapping node ID to waste volume (demand).
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes_set: Set of node IDs that must be serviced.

    Returns:
        List[List[int]]: A filtered list of economically viable routes.
    """
    valid_routes = []

    for route in routes:
        if not route:
            continue

        # 1. Mandatory routes are always kept
        if any(node in mandatory_nodes_set for node in route):
            valid_routes.append(route)
            continue

        # 2. Calculate full route detour cost
        cost = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i], route[i + 1]]
        cost += dist_matrix[route[-1], 0]

        # 3. Calculate total revenue
        revenue = sum(wastes.get(node, 0.0) for node in route) * R

        # 4. Keep if profitable (or effectively break-even to floating point precision)
        profit = revenue - (cost * C)
        if profit >= -1e-4:
            valid_routes.append(route)

    return valid_routes


def greedy_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    blink_rate: float = 0.1,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Standard Greedy insertion with randomized skips ('blinks').
    Optimizes strictly for minimum distance/cost.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        wastes: Node demands.
        capacity: Vehicle capacity.
        blink_rate: Probability of skipping the current best option.
        mandatory_nodes: List of nodes that must be inserted.
        rng: Random number generator.
        expand_pool: If True, reconstructs the unassigned pool from all unvisited nodes.

    Returns:
        Routes with nodes reinserted based on minimum cost.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if rng is None:
        rng = Random(42)

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    # Shuffle for randomness, then stable sort mandatory nodes to the front
    rng.shuffle(unassigned)
    unassigned.sort(key=lambda x: 0 if x in mandatory_set else 1)
    for node in unassigned:
        node_waste = wastes.get(node, 0)
        is_man = node in mandatory_set
        options = []

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                options.append((cost_delta, r_idx, pos))

        # Check new route
        new_route_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        options.append((new_route_cost, len(routes), 0))
        if not options:
            if is_man:
                routes.append([node])
                loads.append(node_waste)
            continue

        # Sort options by Cost (Ascending - Lowest cost is best)
        options.sort(key=lambda x: x[0])

        # True Option Blink Logic
        best_selection = options[-1]  # Fallback to worst
        for opt in options:
            if rng.random() >= blink_rate:
                best_selection = opt
                break

        # Apply the chosen move
        cost, r_idx, pos = best_selection
        if r_idx == len(routes):
            routes.append([node])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, node)
            loads[r_idx] += node_waste

    return routes


def greedy_profit_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    blink_rate: float = 0.1,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Greedy profit-driven insertion with randomized skips ('blinks').
    Includes Speculative Seeding and Economic Pruning for VRPP.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (per waste unit).
        C: Cost multiplier (per distance unit).
        blink_rate: Probability of skipping a valid insertion check.
        mandatory_nodes: List of mandatory node indices.
        rng: Random number generator.
        expand_pool: If True, reconstructs the unassigned pool from all unvisited nodes.

    Returns:
        List[List[int]]: Updated routes, pruned of unprofitable excursions.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if rng is None:
        rng = Random(42)

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    rng.shuffle(unassigned)

    for node in unassigned:
        node_waste = wastes.get(node, 0)
        revenue = node_waste * R
        is_mandatory = node in mandatory_nodes_set

        candidates = []  # List of (profit, route_idx, position)

        # Evaluate existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost_delta * C)

                # Check profitability hurdle
                if is_mandatory or profit > -1e-4:
                    candidates.append((profit, r_idx, pos))

        # Evaluate new route (with speculative seed)
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - (new_cost * C)

        # Allow up to 50% of the return-trip cost to be covered by synergy later.
        seed_hurdle = -0.5 * (new_cost * C)

        if is_mandatory or new_profit >= seed_hurdle:
            candidates.append((new_profit, len(routes), 0))

        # Emergency fallback if no valid options exist
        if not candidates:
            if is_mandatory:
                routes.append([node])
                loads.append(node_waste)
            continue

        # Sort candidates by Profit (Descending - Highest profit is best)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # True Option Blink Logic
        selected = candidates[-1]  # Fallback to worst valid option
        for cand in candidates:
            if rng.random() >= blink_rate:
                selected = cand
                break

        # Apply insertion
        profit, r_idx, pos = selected
        if r_idx == len(routes):
            routes.append([node])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, node)
            loads[r_idx] += node_waste

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
