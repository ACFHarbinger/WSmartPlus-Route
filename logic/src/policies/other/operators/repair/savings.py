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

from typing import Dict, List

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
