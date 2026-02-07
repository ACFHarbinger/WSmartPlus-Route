import random
from typing import Dict, List

import numpy as np


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

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        demands: Demands.
        capacity: Capacity.
        blink_rate: Probability of skipping a check.

    Returns:
        Routes with all nodes reinserted.
    """
    loads = []
    for route in routes:
        loads.append(sum(demands.get(n, 0) for n in route))

    # Reinsert in random order
    random.shuffle(removed_nodes)

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
                # Blink check
                if random.random() < blink_rate:
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
            loads.append(demand)
        else:
            routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += demand

    return routes
