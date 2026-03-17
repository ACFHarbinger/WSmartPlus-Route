"""
Savings-Based Insertion Operator Module.

This module implements a hybrid insertion heuristic for the VRPP. It combines
the Clarke & Wright savings principle with economic profitability checks.
It ensures that mandatory nodes are serves at any cost while opportunistic
nodes are only inserted if their marginal revenue exceeds their detour cost.

Score metric: S = (d(0,u) + d(u,0)) - (d(i,u) + d(u,j) - d(i,j))
"""

from typing import Dict, List, Optional

import numpy as np


def savings_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    Insert removed nodes based on maximum savings versus a dedicated route.

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: Dictionary mapping node ID to waste volume.
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier per kg.
        C: Cost multiplier per km.
        mandatory_nodes: Nodes that must be routed regardless of profit.

    Returns:
        List[List[int]]: Updated routes.
    """
    unassigned = removed_nodes.copy()
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    loads = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    while unassigned:
        best_node = -1
        best_route_idx = -1
        best_pos = -1
        best_saving = float("-inf")

        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            node_waste = wastes.get(node, 0.0)

            # The cost of serving this node alone (depot -> node -> depot)
            dedicated_cost = dist_matrix[0, node] + dist_matrix[node, 0]

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    # Detour cost of inserting 'node' between 'prev' and 'nxt'
                    detour_cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    profit_delta = (node_waste * R) - (detour_cost * C)

                    # VRPP Constraint: Ignore unprofitable insertions for opportunistic nodes
                    if not is_mandatory and profit_delta < -1e-4:
                        continue

                    # Savings: Distance of dedicated route minus the detour distance
                    saving = dedicated_cost - detour_cost

                    # Heavily prioritize mandatory nodes to ensure feasibility
                    effective_saving = saving + (1e9 if is_mandatory else 0)

                    if effective_saving > best_saving:
                        best_saving = effective_saving
                        best_node = node
                        best_route_idx = r_idx
                        best_pos = pos

        if best_node != -1:
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0.0)
            unassigned.remove(best_node)
        else:
            # If no feasible/profitable insertion exists, attempt to open a new route
            # for the remaining mandatory nodes.
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0.0))
                unassigned.remove(node)
            else:
                # Only unprofitable non-mandatory nodes remain. Terminate.
                break

    return routes


def savings_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    depot: int = 0,
) -> List[List[int]]:
    """
    Insert nodes using the Profit-Aware Clarke-Wright (PACW) principle.

    Score metric: S = (d(0,u) + d(u,0)) - (d(i,u) + d(u,j) - d(i,j))
    Constraint: serving node 'u' must yield (w_u * R) - (detour_cost * C) > 0

    Args:
        routes: List of current routes (sequences of node IDs).
        removed_nodes: Nodes waiting to be re-inserted.
        dist_matrix: 2D array of travel distances.
        wastes: Map of node ID to current waste volume (kg).
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier (currency/kg).
        C: Cost multiplier (currency/km).
        mandatory_nodes: Nodes that MUST be served regardless of profit.
        depot: The index representing the depot (default 0).

    Returns:
        List[List[int]]: Solution with nodes re-inserted into routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    unassigned = set(removed_nodes)

    # Pre-calculate loads to avoid repeated sum() calls
    loads = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    while unassigned:
        best_score = float("-inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0.0)
            is_mandatory = node in mandatory_nodes_set
            revenue = node_waste * R

            # Distance of serving this node on a dedicated route from depot
            dedicated_dist = dist_matrix[depot, node] + dist_matrix[node, depot]

            for r_idx, route in enumerate(routes):
                # Capacity Constraint
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else depot
                    nxt = route[pos] if pos < len(route) else depot

                    # Marginal increase in distance
                    detour_dist = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                    # Profit of this specific insertion
                    insertion_profit = revenue - (detour_dist * C)

                    # Economic Termination: Skip if insertion is a net loss (only for non-mandatory)
                    if not is_mandatory and insertion_profit < -1e-4:
                        continue

                    # Savings logic: How much better is this detour than a dedicated route?
                    # S = Dedicated trip distance - Detour distance
                    saving = dedicated_dist - detour_dist

                    # Weighting: Prioritize mandatory nodes via large constant
                    # Weighting: For opportunistic nodes, use distance savings
                    score = saving + (1e9 if is_mandatory else 0)

                    if score > best_score:
                        best_score = score
                        best_node = node
                        best_route_idx = r_idx
                        best_pos = pos

        # Execute insertion if a valid candidate was found
        if best_node != -1:
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0.0)
            unassigned.remove(best_node)
        else:
            # Fallback: Check if we have mandatory nodes that couldn't fit in existing routes
            mandatory_remaining = sorted([n for n in unassigned if n in mandatory_nodes_set])
            if mandatory_remaining:
                # Open a new route for the first remaining mandatory node
                new_node = mandatory_remaining[0]
                routes.append([new_node])
                loads.append(wastes.get(new_node, 0.0))
                unassigned.remove(new_node)
            else:
                # No more profitable opportunistic moves or mandatory nodes exist.
                break

    return routes
