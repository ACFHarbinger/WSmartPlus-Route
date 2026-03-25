"""
Greedy Insertion Operator Module.

This module implements the greedy insertion heuristic, which iteratively inserts
unassigned nodes into the position that minimizes the immediate cost increase.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.greedy import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from typing import Dict, List, Optional

import numpy as np

from ._prune_routes import prune_unprofitable_routes


def greedy_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes into their best (cheapest) positions greedily.

    Iterates through all unassigned nodes and all possible insertion positions,
    finding the globally cheapest insertion and applying it. Repeats until all
    nodes are inserted OR skipping occurs based on profitability (VRPP).

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Noise level for cost perturbation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    # Calculate current loads and track visited nodes
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))  # Sort for deterministic ties

    while unassigned:
        best_cost = float("inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)

            for i, route in enumerate(routes):
                if loads[i] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    # Cost increase: d(i-1, node) + d(node, i) - d(i-1, i)
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    # Calculate base insertion cost
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                    # Apply noise additively (Ropke & Pisinger 2005, Section 3.4.3)
                    # C' = max{0, C + noise} where noise is a perturbation
                    if noise != 0:
                        cost = max(0.0, cost + noise)

                    if cost < best_cost:
                        best_cost = cost
                        best_node = node
                        best_route_idx = i
                        best_pos = pos

        if best_node != -1:
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # Check for remaining mandatory nodes
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
            else:
                # No more mandatory nodes can be inserted.
                # To prevent infinite loops, we must clear unassigned if no more feasible insertions exist.
                break

    return routes


def greedy_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Greedily insert nodes to maximize profit (revenue - cost).

    VRPP logic: Instead of minimizing cost, we maximize (waste * R - delta_dist * C).
    Nodes with negative max profit are skipped unless they are mandatory.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (per waste unit).
        C: Cost multiplier (per distance unit).
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, all unvisited nodes are considered.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0) for node in r) for r in routes]

    visited = set()
    for r in routes:
        visited.update(r)

    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        best_profit = -float("inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)
            revenue = node_waste * R
            is_mandatory = node in mandatory_nodes_set

            for i, route in enumerate(routes):
                if loads[i] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    # Cost increase: d(prev, node) + d(node, nxt) - d(prev, nxt)
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    profit = revenue - (cost * C)

                    # If mandatory, we should prioritize insertion even if profit is negative?
                    # A common approach is to add a large constant to profit for mandatory nodes
                    # or just ensure they are inserted.
                    effective_profit = profit + (1e9 if is_mandatory else 0)

                    if effective_profit > best_profit:
                        # Even for non-mandatory, we only insert if profit > 0 (or some small epsilon)
                        if not is_mandatory and profit < -1e-4:
                            continue

                        best_profit = effective_profit
                        best_node = node
                        best_route_idx = i
                        best_pos = pos

            # Evaluate new route (Speculative Seeding Heuristic)
            # Theoretical justification: Lower-bound expectation on route profitability
            #
            # A new route starting with a single node has immediate profit:
            #   π₀ = revenue - cost = (waste * R) - (2 * d_{0,node} * C)
            #
            # The hurdle π_hurdle = -0.5 * (2 * d_{0,node} * C) allows seeding routes
            # that are initially unprofitable but likely to become profitable when
            # additional profitable nodes are inserted later. This threshold represents
            # a speculative investment based on the expected value of future insertions.
            #
            # Empirically, this enables exploration of promising partial solutions that
            # would otherwise be rejected by pure greedy profit maximization.
            new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
            new_profit = revenue - (new_cost * C)
            seed_hurdle = -0.5 * (new_cost * C)  # Speculative hurdle: 50% of detour cost

            if is_mandatory or new_profit >= seed_hurdle:
                new_effective_profit = new_profit + (1e9 if is_mandatory else 0)
                if new_effective_profit > best_profit:
                    best_profit = new_effective_profit
                    best_node = node
                    best_route_idx = len(routes)
                    best_pos = 0

        if best_node != -1:
            if best_route_idx == len(routes):
                routes.append([best_node])
                loads.append(wastes.get(best_node, 0))
            else:
                routes[best_route_idx].insert(best_pos, best_node)
                loads[best_route_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # Handle remaining mandatory nodes by opening new routes if possible
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
            else:
                break

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
