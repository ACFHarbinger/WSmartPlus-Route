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
        demands (Dict[int, float]): Demand look-up.
        capacity (float): Vehicle capacity.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    # Calculate current loads
    loads = []
    for route in routes:
        loads.append(sum(demands.get(n, 0) for n in route))

    # Shuffle removed nodes to avoid deterministic bias
    # But for pure greedy, order matters. ALNS usually randomizes slightly.
    # Here we stick to input order or simple iteration.

    for node in removed_nodes:
        demand = demands.get(node, 0)
        best_cost = float("inf")
        best_r_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + demand > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = 0 if pos == 0 else route[pos - 1]
                nex = 0 if pos == len(route) else route[pos]

                cost = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]

                if cost < best_cost:
                    best_cost = cost
                    best_r_idx = r_idx
                    best_pos = pos

        # Check new route
        # Cost is 0 -> node -> 0
        new_route_cost = dist_matrix[0][node] + dist_matrix[node][0]
        if new_route_cost < best_cost:
            best_cost = new_route_cost
            best_r_idx = len(routes)
            best_pos = 0

        # Apply insertion
        if best_r_idx == len(routes):
            routes.append([node])
            loads.append(demand)
        else:
            routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += demand

    return routes
