"""
Savings-based Insertion Operator Module.

Reinserts unassigned nodes based on the Clarke & Wright savings principle:

    S_ij = d(i, depot) + d(depot, j) - d(i, j)

Nodes that maximise this saving are inserted into positions that merge
trips, respecting capacity constraints.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.savings import savings_insertion
    >>> routes = savings_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from typing import Dict, List, Optional

import numpy as np


def savings_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    depot: int = 0,
) -> List[List[int]]:
    """
    Insert removed nodes based on Clarke-Wright savings.

    For each unassigned node, the savings value for placing it between
    every pair of adjacent nodes in existing routes is computed.  The
    globally best (highest saving) feasible insertion is applied first.

    Args:
        routes: Partial routes (may be empty or partially filled).
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix ``(N+1, N+1)``.
        wastes: Waste/demand look-up per node.
        capacity: Vehicle capacity.
        depot: Depot index (default 0).

    Returns:
        Updated routes with nodes inserted.
    """
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    unassigned = sorted(list(removed_nodes))

    while unassigned:
        best_saving = -float("inf")
        best_node = -1
        best_route = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else depot
                    nxt = route[pos] if pos < len(route) else depot

                    # Savings: cost of separate depot trips minus merged cost
                    separate = dist_matrix[prev, depot] + dist_matrix[depot, nxt]
                    merged = dist_matrix[prev, node] + dist_matrix[node, nxt]
                    saving = separate - merged

                    if saving > best_saving:
                        best_saving = saving
                        best_node = node
                        best_route = r_idx
                        best_pos = pos

        if best_node != -1:
            routes[best_route].insert(best_pos, best_node)
            loads[best_route] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # Create new routes for remaining nodes
            for node in unassigned:
                routes.append([node])
                loads.append(wastes.get(node, 0))
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
    Savings-based insertion for VRPP.

    Calculates the profit gain for each insertion and prioritizes those
    with the highest 'profit-saving' (considering both revenue and distance).

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned nodes.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: List of mandatory node indices.
        depot: Depot index.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    unassigned = sorted(list(removed_nodes))

    while unassigned:
        best_profit_gain = -float("inf")
        best_node = -1
        best_route = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)
            revenue = node_waste * R
            is_mandatory = node in mandatory_nodes_set

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else depot
                    nxt = route[pos] if pos < len(route) else depot

                    # Traditional saving logic: S = d(i,0) + d(0,j) - d(i,j)
                    # For profit, we consider the net gain: Revenue - CostIncrease
                    # where CostIncrease = d(i,u) + d(u,j) - d(i,j)
                    cost_increase = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    profit_gain = revenue - (cost_increase * C)

                    effective_gain = profit_gain + (1e9 if is_mandatory else 0)

                    if effective_gain > best_profit_gain:
                        if not is_mandatory and profit_gain < -1e-4:
                            continue
                        best_profit_gain = effective_gain
                        best_node = node
                        best_route = r_idx
                        best_pos = pos

        if best_node != -1:
            routes[best_route].insert(best_pos, best_node)
            loads[best_route] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # Handle remaining mandatory nodes
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                continue
            else:
                break

    return routes
