"""
Greedy Initialization Module.

Creates an initial constructive solution for the VRPP using greedy insertion,
enforcing strict economic termination to drop unprofitable opportunistic nodes.
"""

import random
from typing import Dict, List, Optional

import numpy as np


def build_greedy_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """
    Create an initial solution using a simple greedy heuristic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Waste dictionary mapping node ID to waste volume.
        capacity: Vehicle capacity.
        R: Revenue per unit.
        C: Cost per distance unit.
        mandatory_nodes: List of nodes that MUST be visited.
        rng: Random number generator.

    Returns:
        List[List[int]]: Initial routing solution.
    """
    if rng is None:
        rng = random.Random(42)

    n_nodes = len(dist_matrix) - 1
    # Nodes are assumed to be 1-indexed (0 is depot)
    nodes = list(range(1, n_nodes + 1))

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    unvisited = set(nodes)

    routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0

    while unvisited:
        last_node = 0 if len(current_route) == 0 else current_route[-1]

        best_node = None
        best_profit = float("-inf")

        # Evaluate all unvisited nodes for the next step in the current route
        for node in unvisited:
            node_w = wastes.get(node, 0.0)
            if current_load + node_w <= capacity:
                node_revenue = node_w * R
                distance = dist_matrix[last_node, node]
                profit = node_revenue - (distance * C)

                if profit > best_profit:
                    best_profit = profit
                    best_node = node

        if best_node is not None:
            # STRICT ECONOMIC TERMINATION CONDITION
            # If the best available node results in a financial loss and is not mandatory,
            # drop it from the pool permanently to prevent infinite loops of bad routes.
            if best_profit < 0 and best_node not in mandatory_nodes_set:
                unvisited.remove(best_node)
                continue

            # Add node to current route
            current_route.append(best_node)
            current_load += wastes.get(best_node, 0.0)
            unvisited.remove(best_node)
        else:
            # Cannot add any more nodes to the current route (due to capacity)
            if len(current_route) > 0:
                routes.append(current_route)

            current_route = []
            current_load = 0.0

            # If we still have unvisited nodes, check if any fit in an empty truck
            if unvisited:
                still_constructible = False
                for node in list(unvisited):
                    if wastes.get(node, 0.0) <= capacity:
                        still_constructible = True
                    else:
                        unvisited.remove(node)  # Node exceeds vehicle capacity entirely

                if not still_constructible:
                    break
                continue
            else:
                break

    # Add the final route if it contains nodes
    if len(current_route) > 0:
        routes.append(current_route)

    return routes
