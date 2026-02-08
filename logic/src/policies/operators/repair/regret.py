"""
Regret Insertion Operator Module.

This module implements the Regret-k insertion heuristic for VRP repair.
It calculates a regret value for each unassigned node based on the cost difference
between its best and k-th best insertion positions.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.repair.regret import RegretInsertion
    >>> operator = RegretInsertion(k=2)
    >>> new_routes = operator.repair(destroyed_routes, unassigned_nodes)
"""

from typing import Dict, List

import numpy as np


def regret_2_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
) -> List[List[int]]:
    """
    Insert removed nodes based on the regret-2 criterion.
    Regret-2 Insertion Heuristic.

    Prioritizes inserting nodes that would be much more expensive to insert
    later (high regret = cost difference between best and 2nd best option).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        demands: Demand look-up.
        capacity: Vehicle capacity.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(routes, removed_nodes, dist_matrix, demands, capacity, k=2)


def regret_k_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    k: int = 2,
) -> List[List[int]]:
    """
    Regret-k Insertion Heuristic.

    Generalization of regret insertion to k-th best option.
    Calculates regret as cost(k-th best) - cost(best).

    For each unassigned node, calculate the difference between the best and
    k-th best insertion cost. Insert the node with maximum regret first.

    Args:
        routes: Partial routes.
        demands: Node demands.
        capacity: Vehicle capacity.
        k: Regret degree (2, 3, etc.).

    Returns:
        Routes with all nodes inserted.
    """
    # Initialize loads
    loads = []
    for route in routes:
        loads.append(sum(demands.get(n, 0) for n in route))

    unassigned = list(removed_nodes)

    while unassigned:
        # Calculate insertion costs for all unassigned nodes in all positions
        # node -> [(cost, r_idx, pos), ...]
        all_candidates = []

        for node in unassigned:
            demand = demands.get(node, 0)
            node_options = []

            # Check existing routes
            for r_idx, route in enumerate(routes):
                if loads[r_idx] + demand > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = 0 if pos == 0 else route[pos - 1]
                    nex = 0 if pos == len(route) else route[pos]

                    cost = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]
                    node_options.append((cost, r_idx, pos))

            # New route option
            new_cost = dist_matrix[0][node] + dist_matrix[node][0]
            node_options.append((new_cost, len(routes), 0))

            # Sort options by cost
            node_options.sort(key=lambda x: x[0])

            # Calculate regret
            if len(node_options) >= k:
                # Regret = cost_at_k - cost_at_1
                regret = node_options[k - 1][0] - node_options[0][0]
            elif len(node_options) > 1:
                # If fewer than k options, regret is diff between last and first
                regret = node_options[-1][0] - node_options[0][0]
            else:
                # Only one option (or none), max priority
                regret = float("inf")

            best_option = node_options[0] if node_options else (float("inf"), -1, -1)
            all_candidates.append((regret, node, best_option))

        if not all_candidates:
            # Should not happen if feasible
            break

        # Pick node with max regret
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_node, (cost, r_idx, pos) = all_candidates[0]

        if r_idx == -1:
            # Cannot insert node anywhere
            break

        # Apply insertion
        demand = demands.get(best_node, 0)
        if r_idx == len(routes):
            routes.append([best_node])
            loads.append(demand)
        else:
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += demand

        unassigned.remove(best_node)

    return routes
