"""
GENI (Generalized Insertion) Operator Module.

Implements GENI Type I and Type II moves that insert a node between two
*non-adjacent* nodes in the tour, followed by localized edge re-optimization.

- **Type I**: Insert node ``u`` between non-adjacent ``v_i`` and ``v_j``,
  reconnecting with a single bypass.
- **Type II**: Insert node ``u`` between ``v_i`` and ``v_j`` while
  simultaneously reversing the segment between them.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.geni import geni_insertion
    >>> improved = geni_insertion(ls, node=5, r_idx=0, neighborhood_size=5)
"""

from typing import Any, List, Tuple

import numpy as np


def geni_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: dict,
    capacity: float,
    R: float,
    C: float,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """
    GENI insertion: insert removed nodes into routes using Type I and II moves.

    Args:
        routes: List of current routes.
        removed_nodes: Nodes to be inserted.
        dist_matrix: Distance matrix.
        wastes: Dictionary of node wastes (demands).
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        neighborhood_size: Number of closest nodes in route to consider as v_i.

    Returns:
        List[List[int]]: Updated routes.
    """
    nodes_to_process = list(removed_nodes)

    while nodes_to_process:
        best_profit = -1e-4
        best_move = None  # (node, route_idx, i, j, type)

        for node in nodes_to_process:
            revenue = wastes.get(node, 0) * R
            node_waste = wastes.get(node, 0)

            # 1. Evaluate insertion into existing routes
            for r_idx, route in enumerate(routes):
                profit, move = _evaluate_route_insertion(
                    node, r_idx, route, dist_matrix, wastes, capacity, revenue, node_waste, C, neighborhood_size
                )
                if profit > best_profit:
                    best_profit = profit
                    best_move = move

            # 2. Evaluate opening a new route
            delta_new = dist_matrix[0, node] + dist_matrix[node, 0]
            profit_new = revenue - (delta_new * C)
            if profit_new > best_profit:
                best_profit = profit_new
                best_move = (node, -1, -1, -1, "NEW")

        if best_move:
            u, r_idx, i, j, m_type = best_move
            nodes_to_process.remove(u)
            _apply_geni_move(routes, u, r_idx, i, j, m_type)
        else:
            break

    return routes


def _evaluate_route_insertion(
    node: int,
    r_idx: int,
    route: List[int],
    dist_matrix: np.ndarray,
    wastes: dict,
    capacity: float,
    revenue: float,
    node_waste: float,
    C: float,
    neighborhood_size: int,
) -> Tuple[float, Any]:
    """Helper to evaluate Type I and Type II insertions into a specific route."""
    current_load = sum(wastes.get(n, 0) for n in route)
    if current_load + node_waste > capacity:
        return -float("inf"), None

    if not route:
        delta = dist_matrix[0, node] + dist_matrix[node, 0]
        return revenue - (delta * C), (node, r_idx, -1, -1, "SIMPLE")

    best_profit = -float("inf")
    best_move = None

    potential_vi_indices = sorted(range(len(route)), key=lambda idx: dist_matrix[route[idx], node])[:neighborhood_size]

    for i in potential_vi_indices:
        v_i = route[i]
        v_i_next = route[i + 1] if i + 1 < len(route) else 0

        for j in range(len(route)):
            if j == i or j == i + 1 or (i == len(route) - 1 and j == 0):
                continue

            v_j = route[j]
            v_j_next = route[j + 1] if j + 1 < len(route) else 0
            v_j_prev = route[j - 1] if j > 0 else 0

            # Type I: j > i
            if j > i:
                delta_i = (
                    dist_matrix[v_i, node]
                    + dist_matrix[node, v_j]
                    + dist_matrix[v_i_next, v_j_next]
                    - (dist_matrix[v_i, v_i_next] + dist_matrix[v_j, v_j_next])
                )
                profit_i = revenue - (delta_i * C)
                if profit_i > best_profit:
                    best_profit = profit_i
                    best_move = (node, r_idx, i, j, "TYPE_I")

            # Type II: j > i + 1
            if j > i + 1:
                delta_ii = (
                    dist_matrix[v_i, node]
                    + dist_matrix[node, v_j_prev]
                    + dist_matrix[v_i_next, v_j]
                    - (dist_matrix[v_i, v_i_next] + dist_matrix[v_j_prev, v_j])
                )
                profit_ii = revenue - (delta_ii * C)
                if profit_ii > best_profit:
                    best_profit = profit_ii
                    best_move = (node, r_idx, i, j, "TYPE_II")

    return best_profit, best_move


def _apply_geni_move(routes: List[List[int]], u: int, r_idx: int, i: int, j: int, m_type: str):
    """Helper to apply the chosen GENI move to the routes list."""
    if m_type == "SIMPLE":
        routes[r_idx] = [u]
    elif m_type == "NEW":
        routes.append([u])
    elif m_type == "TYPE_I":
        route = routes[r_idx]
        routes[r_idx] = route[: i + 1] + [u] + route[i + 1 : j + 1][::-1] + route[j + 1 :]
    elif m_type == "TYPE_II":
        route = routes[r_idx]
        routes[r_idx] = route[: i + 1] + [u] + route[i + 1 : j][::-1] + route[j:]
