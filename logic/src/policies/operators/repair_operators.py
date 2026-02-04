"""
Repair operators for the Adaptive Large Neighborhood Search (ALNS).

This module contains various insertion heuristics used to re-integrate
removed nodes back into the routing solution.
"""

import random
from typing import Dict, List

import numpy as np


def greedy_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
) -> List[List[int]]:
    """
    Insert removed nodes into their best (cheapest) positions greedily.

    Args:
        routes (List[List[int]]): Partial routes.
        removed_nodes (List[int]): Nodes to be re-inserted.
        dist_matrix (np.ndarray): Distance matrix.
        demands (Dict[int, float]): Node demands.
        capacity (float): Vehicle capacity.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    # Insert each node in best position
    random.shuffle(removed_nodes)  # Randomize order

    for node in removed_nodes:
        best_cost = float("inf")
        best_pos = None  # (route_idx, insert_idx)

        # Try all positions
        for r_idx, route in enumerate(routes):
            load = sum(demands.get(n, 0) for n in route)
            if load + demands.get(node, 0) > capacity:
                continue

            # Try all slots: 0 to len(route)
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) else route[i]

                cost_increase = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]

                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_pos = (r_idx, i)

        # Also consider new route
        if demands.get(node, 0) <= capacity:
            cost_new = dist_matrix[0][node] + dist_matrix[node][0]
            if cost_new < best_cost:
                best_cost = cost_new
                best_pos = (len(routes), 0)

        # Apply
        if best_pos:
            r, m = best_pos
            if r == len(routes):
                routes.append([node])
            else:
                routes[r].insert(m, node)
        else:
            routes.append([node])

    return routes


def regret_2_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
) -> List[List[int]]:
    """
    Insert removed nodes based on the regret-2 criterion.

    Args:
        routes (List[List[int]]): Partial routes.
        removed_nodes (List[int]): Nodes to be re-inserted.
        dist_matrix (np.ndarray): Distance matrix.
        demands (Dict[int, float]): Node demands.
        capacity (float): Vehicle capacity.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    if len(removed_nodes) > 30:
        return greedy_insertion(routes, removed_nodes, dist_matrix, demands, capacity)

    pending = removed_nodes[:]

    while pending:
        max_regret = -1
        best_node_to_insert = None
        best_insert_pos = None  # (r, i)

        # For each node, find Best and 2nd Best insertion scores
        for node in pending:
            valid_moves = []  # (cost_increase, r, i)

            # Check existing routes
            for r_idx, route in enumerate(routes):
                load = sum(demands.get(n, 0) for n in route)
                if load + demands.get(node, 0) > capacity:
                    continue

                for i in range(len(route) + 1):
                    prev = 0 if i == 0 else route[i - 1]
                    nex = 0 if i == len(route) else route[i]
                    cost = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]
                    valid_moves.append((cost, r_idx, i))

            # Check new route
            if demands.get(node, 0) <= capacity:
                cost = dist_matrix[0][node] + dist_matrix[node][0]
                valid_moves.append((cost, len(routes), 0))

            if not valid_moves:
                valid_moves.append((dist_matrix[0][node] * 2, len(routes), 0))

            valid_moves.sort(key=lambda x: x[0])

            best = valid_moves[0]
            second = valid_moves[1] if len(valid_moves) > 1 else (best[0] * 1.5, -1, -1)

            regret = second[0] - best[0]

            if regret > max_regret:
                max_regret = regret
                best_node_to_insert = node
                best_insert_pos = (best[1], best[2])

        # Apply Best Regret Move
        if best_node_to_insert:
            assert best_insert_pos is not None
            r, m = best_insert_pos
            if r == len(routes):
                routes.append([best_node_to_insert])
            else:
                routes[r].insert(m, best_node_to_insert)
            pending.remove(best_node_to_insert)
        else:
            break

    return routes
