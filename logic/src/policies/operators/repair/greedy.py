"""
Greedy Insertion Operator Module.

This module implements the greedy insertion heuristic, which iteratively inserts
unassigned nodes into the position that minimizes the immediate cost increase.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.repair.greedy import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, demands, capacity)
"""

from typing import Dict, List, Optional

import numpy as np


def greedy_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    R: Optional[float] = None,
) -> List[List[int]]:
    """
    Insert removed nodes into their best (cheapest) positions greedily.

    Iterates through all unassigned nodes and all possible insertion positions,
    finding the globally cheapest insertion and applying it. Repeats until all
    nodes are inserted.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        demands: Demand look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (Optional). If provided, insertion is skipped if cost > revenue.

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

        # VRPP Profit Check
        if R is not None:
            revenue = demand * R
            if best_cost > revenue:
                continue

        # Apply insertion
        if best_r_idx == len(routes):
            routes.append([node])
            loads.append(demand)
        else:
            routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += demand

    return routes
