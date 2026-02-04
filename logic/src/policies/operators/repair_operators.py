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


def regret_k_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    k: int = 3,
) -> List[List[int]]:
    """
    Generalized Regret-k insertion heuristic.

    For each unassigned node, calculate the difference between the best and
    k-th best insertion cost. Insert the node with maximum regret first.
    This look-ahead helps avoid myopic insertions.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        demands: Node demand dictionary.
        capacity: Vehicle capacity.
        k: Number of routes to consider for regret (typically 2-4).

    Returns:
        Routes with all nodes inserted.
    """
    # Fall back to greedy for large instances
    if len(removed_nodes) > 40:
        return greedy_insertion(routes, removed_nodes, dist_matrix, demands, capacity)

    pending = removed_nodes[:]

    while pending:
        max_regret = float("-inf")
        best_node_to_insert = None
        best_insert_pos = None

        for node in pending:
            valid_moves = []  # (cost, route_idx, position)

            node_demand = demands.get(node, 0)

            # Evaluate all positions in existing routes
            for r_idx, route in enumerate(routes):
                load = sum(demands.get(n, 0) for n in route)
                if load + node_demand > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = 0 if pos == 0 else route[pos - 1]
                    nxt = 0 if pos == len(route) else route[pos]
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    valid_moves.append((cost, r_idx, pos))

            # Consider new route
            if node_demand <= capacity:
                new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
                valid_moves.append((new_cost, len(routes), 0))

            if not valid_moves:
                # Fallback
                valid_moves.append((float("inf"), len(routes), 0))

            valid_moves.sort(key=lambda x: x[0])

            # Calculate regret: difference between best and k-th best
            best_cost = valid_moves[0][0]
            kth_idx = min(k - 1, len(valid_moves) - 1)
            kth_cost = valid_moves[kth_idx][0]

            regret = kth_cost - best_cost

            if regret > max_regret:
                max_regret = regret
                best_node_to_insert = node
                best_insert_pos = (valid_moves[0][1], valid_moves[0][2])

        # Apply best regret move
        if best_node_to_insert is not None and best_insert_pos is not None:
            r, m = best_insert_pos
            if r >= len(routes):
                routes.append([best_node_to_insert])
            else:
                routes[r].insert(m, best_node_to_insert)
            pending.remove(best_node_to_insert)
        else:
            break

    return routes


def greedy_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    blink_rate: float = 0.2,
) -> List[List[int]]:
    """
    Greedy insertion with probabilistic "blinks" for speed.

    SISR uses this fast insertion operator to allow millions of iterations.
    With probability `blink_rate`, skip checking a position entirely.
    This randomizes the insertion while maintaining speed.

    Args:
        routes: Partial routes after destruction.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        demands: Node demand dictionary.
        capacity: Vehicle capacity.
        blink_rate: Probability of skipping a position check.

    Returns:
        Routes with all nodes reinserted.
    """
    random.shuffle(removed_nodes)

    for node in removed_nodes:
        best_cost = float("inf")
        best_pos = None  # (route_idx, insert_pos)

        node_demand = demands.get(node, 0)

        for r_idx, route in enumerate(routes):
            load = sum(demands.get(n, 0) for n in route)
            if load + node_demand > capacity:
                continue

            for i in range(len(route) + 1):
                # Blink: skip this position with probability blink_rate
                if random.random() < blink_rate:
                    continue

                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]

                cost_increase = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                if cost_increase < best_cost:
                    best_cost = cost_increase
                    best_pos = (r_idx, i)

        # Consider new route
        if node_demand <= capacity:
            new_route_cost = dist_matrix[0, node] + dist_matrix[node, 0]
            if new_route_cost < best_cost:
                best_cost = new_route_cost
                best_pos = (len(routes), 0)

        # Apply insertion
        if best_pos:
            r, m = best_pos
            if r == len(routes):
                routes.append([node])
            else:
                routes[r].insert(m, node)
        else:
            # Fallback: create new route
            routes.append([node])

    return routes
