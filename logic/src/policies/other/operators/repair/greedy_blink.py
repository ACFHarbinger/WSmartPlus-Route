"""
Greedy Blink Insertion Operator Module.

This module implements the 'Greedy with Blinks' insertion heuristic. It is a
randomized version of greedy insertion where the algorithm 'blinks' (skips)
the best position with some probability, encouraging diversity.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.greedy_blink import greedy_insertion_with_blinks
    >>> routes = greedy_insertion_with_blinks(routes, removed, dist_matrix, wastes, capacity, blink_rate=0.1)
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np


def greedy_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    blink_rate: float = 0.1,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Greedy insertion with randomized skips ('blinks').

    Similar to standard greedy insertion, but during the selection of the best
    position, the algorithm may skip the absolute best option with probability
    `blink_rate` and choose the second best, and so on.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        wastes: wastes.
        capacity: Capacity.
        blink_rate: Probability of skipping a check.
        rng: Random number generator.

    Returns:
        Routes with all nodes reinserted.
    """
    loads = []
    for route in routes:
        loads.append(sum(wastes.get(n, 0) for n in route))

    if rng is None:
        rng = Random(42)

    # Reinsert in random order
    rng.shuffle(removed_nodes)

    for node in removed_nodes:
        waste = wastes.get(node, 0)
        best_cost = float("inf")
        best_r_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + waste > capacity:
                continue

            for pos in range(len(route) + 1):
                # Blink check
                if rng.random() < blink_rate:
                    continue

                prev = 0 if pos == 0 else route[pos - 1]
                nex = 0 if pos == len(route) else route[pos]

                cost = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]

                if cost < best_cost:
                    best_cost = cost
                    best_r_idx = r_idx
                    best_pos = pos

        # Check new route
        new_route_cost = dist_matrix[0][node] + dist_matrix[node][0]
        if new_route_cost < best_cost:
            best_cost = new_route_cost
            best_r_idx = len(routes)
            best_pos = 0

        # Apply
        if best_r_idx == len(routes):
            routes.append([node])
            loads.append(waste)
        else:
            routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += waste

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
) -> List[List[int]]:
    """
    Greedy profit-driven insertion with randomized skips ('blinks').

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

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if rng is None:
        rng = Random(42)

    # Reinsert in random order to increase diversity
    unassigned = sorted(list(removed_nodes))  # Stabilize
    rng.shuffle(unassigned)

    for node in unassigned:
        node_waste = wastes.get(node, 0)
        revenue = node_waste * R
        is_mandatory = node in mandatory_nodes_set

        best_profit = -float("inf")
        best_r_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                # Blink check: skip if RNG allows
                if rng.random() < blink_rate:
                    continue

                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost * C)

                if profit > best_profit:
                    if not is_mandatory and profit < -1e-4:
                        continue
                    best_profit = profit
                    best_r_idx = r_idx
                    best_pos = pos

        # Check new route option
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - (new_cost * C)
        if new_profit > best_profit and (is_mandatory or new_profit >= -1e-4):
            best_profit = new_profit
            best_r_idx = len(routes)
            best_pos = 0

        # Apply insertion if found
        if best_r_idx != -1:
            if best_r_idx == len(routes):
                routes.append([node])
                loads.append(node_waste)
            else:
                routes[best_r_idx].insert(best_pos, node)
                loads[best_r_idx] += node_waste
        elif is_mandatory:
            # Must insert mandatory nodes, even if best option was blinked away or unprofitable
            # Prioritize opening a new route if no valid position was found/accepted
            routes.append([node])
            loads.append(node_waste)

    return routes
