"""
Farthest Insertion Repair Operator Module.

This module implements the farthest insertion heuristic, a classical TSP construction
method adapted for VRP. It iteratively selects the unassigned node farthest from the
current tour and inserts it at the position that minimizes the cost increase.

This heuristic is particularly effective for creating geographically diverse tours
and avoiding local clustering, which can be beneficial for VRPP instances.

Reference:
    Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977).
    "An analysis of several heuristics for the traveling salesman problem".
    SIAM Journal on Computing, 6(3), 563-581.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.farthest import farthest_insertion
    >>> routes = farthest_insertion(routes, removed, dist_matrix, wastes, capacity, R, C)
"""

from typing import Dict, List, Optional

import numpy as np


def farthest_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: Optional[float] = None,
    C: Optional[float] = None,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
) -> List[List[int]]:
    """
    Insert removed nodes using the farthest insertion heuristic.

    The algorithm works as follows:
    1. For each unassigned node, compute its minimum distance to any node in any route
    2. Select the node with the maximum such distance (farthest node)
    3. Insert it at the position that minimizes cost increase (cheapest insertion)
    4. Repeat until all nodes are inserted or profitability constraints prevent further insertion

    This creates geographically diverse tours and helps avoid premature clustering.

    Args:
        routes: Partial routes (list of node sequences).
        removed_nodes: List of unassigned node indices.
        dist_matrix: Symmetric distance matrix (N+1 x N+1, including depot at index 0).
        wastes: Dictionary mapping node index to waste/demand.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit waste (optional, for VRPP profitability checks).
        C: Cost per unit distance (optional, for VRPP profitability checks).
        mandatory_nodes: List of mandatory node indices that must be visited.
        expand_pool: If True, consider all unvisited nodes; if False, only removed_nodes.

    Returns:
        List[List[int]]: Updated routes after farthest insertion.

    Note:
        - If R and C are provided, nodes are only inserted if profitable (revenue > cost)
          unless they are mandatory.
        - If a node cannot be feasibly inserted into any route (capacity violation),
          a new route is created for mandatory nodes.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Calculate current loads and track visited nodes
    loads = [sum(wastes.get(node, 0.0) for node in route) for route in routes]
    visited = set()
    for route in routes:
        visited.update(route)

    # Determine candidate pool
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        # Step 1: Find the farthest unassigned node from any route
        farthest_node = None
        max_min_distance = -1.0

        for node in unassigned:
            # Compute minimum distance from this node to any node in any route
            min_distance = float("inf")

            for route in routes:
                for route_node in route:
                    distance = dist_matrix[node, route_node]
                    min_distance = min(min_distance, distance)

            # Also consider distance to depot (for empty routes or depot-only routes)
            depot_distance = dist_matrix[node, 0]
            min_distance = min(min_distance, depot_distance)

            # Track node with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_node = node

        if farthest_node is None:
            break

        # Step 2: Find cheapest insertion position for the farthest node
        node_waste = wastes.get(farthest_node, 0.0)
        revenue = node_waste * R if R is not None else float("inf")
        is_mandatory = farthest_node in mandatory_set

        best_cost = float("inf")
        best_route_idx = -1
        best_pos = -1

        for i, route in enumerate(routes):
            # Check capacity feasibility
            if loads[i] + node_waste > capacity:
                continue

            # Try all insertion positions
            for pos in range(len(route) + 1):
                # Compute cost increase: d(prev, node) + d(node, next) - d(prev, next)
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_increase = (
                    dist_matrix[prev, farthest_node] + dist_matrix[farthest_node, nxt] - dist_matrix[prev, nxt]
                )

                # VRPP profitability check (if R and C are provided)
                if R is not None and C is not None:
                    insertion_cost = cost_increase * C
                    if not is_mandatory and insertion_cost > revenue:
                        continue  # Skip unprofitable insertions

                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_route_idx = i
                    best_pos = pos

        # Step 3: Insert the farthest node at the best position
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_pos, farthest_node)
            loads[best_route_idx] += node_waste
            unassigned.remove(farthest_node)
        else:
            # No feasible insertion found
            # If mandatory, create a new route
            if is_mandatory:
                routes.append([farthest_node])
                loads.append(node_waste)
                unassigned.remove(farthest_node)
            else:
                # For optional nodes, if no profitable/feasible insertion exists, skip it
                unassigned.remove(farthest_node)

    return routes


def farthest_profit_insertion(
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
    Farthest insertion with explicit profit maximization for VRPP.

    Similar to farthest_insertion, but selects the farthest node and then
    inserts it at the position that maximizes profit (revenue - cost) rather
    than minimizing cost alone.

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: Waste/demand dictionary.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory_nodes: Mandatory node indices.
        expand_pool: Consider all unvisited nodes if True.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0.0) for node in route) for route in routes]

    visited = set()
    for route in routes:
        visited.update(route)

    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        # Find farthest node
        farthest_node = None
        max_min_distance = -1.0

        for node in unassigned:
            min_distance = float("inf")
            for route in routes:
                for route_node in route:
                    min_distance = min(min_distance, dist_matrix[node, route_node])
            min_distance = min(min_distance, dist_matrix[node, 0])

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_node = node

        if farthest_node is None:
            break

        # Find best insertion position based on profit
        node_waste = wastes.get(farthest_node, 0.0)
        revenue = node_waste * R
        is_mandatory = farthest_node in mandatory_set

        best_profit = -float("inf")
        best_route_idx = -1
        best_pos = -1

        for i, route in enumerate(routes):
            if loads[i] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_increase = (
                    dist_matrix[prev, farthest_node] + dist_matrix[farthest_node, nxt] - dist_matrix[prev, nxt]
                )
                profit = revenue - (cost_increase * C)

                # Boost mandatory nodes
                effective_profit = profit + (1e9 if is_mandatory else 0)

                if effective_profit > best_profit:
                    if not is_mandatory and profit < -1e-4:
                        continue
                    best_profit = effective_profit
                    best_route_idx = i
                    best_pos = pos

        # Insert at best position
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_pos, farthest_node)
            loads[best_route_idx] += node_waste
            unassigned.remove(farthest_node)
        else:
            # Handle mandatory nodes
            if is_mandatory:
                routes.append([farthest_node])
                loads.append(node_waste)
                unassigned.remove(farthest_node)
            else:
                # Skip unprofitable optional node
                unassigned.remove(farthest_node)

    return routes
