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
        wastes: Waste dictionary.
        capacity: Vehicle capacity.
        R: Revenue per unit.
        C: Cost per distance unit.
        mandatory_nodes: List of nodes that MUST be visited.
        rng: Random number generator.

    Returns:
        Initial solution.
    """
    if rng is None:
        rng = random.Random(42)

    n_nodes = len(dist_matrix) - 1
    nodes = list(range(1, n_nodes + 1))

    # Ensure mandatory nodes are visited
    unvisited = set(mandatory_nodes) if mandatory_nodes else set(nodes)

    routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0

    # Greedy construction: feasible node with best profit
    while unvisited:
        last_node = 0 if len(current_route) == 0 else current_route[-1]

        # Find feasible node with best profit
        best_node = None
        best_profit = float("-inf")
        for node in unvisited:
            node_w = wastes.get(node, 0.0)
            if current_load + node_w <= capacity:
                node_revenue = node_w * R
                distance = dist_matrix[last_node, node] * C
                profit = node_revenue - distance
                if profit > best_profit:
                    best_profit = profit
                    best_node = node

        if best_node is not None:
            # Add node to current route
            current_route.append(best_node)
            current_load += wastes.get(best_node, 0.0)
            unvisited.remove(best_node)
        else:
            # Cannot add any more nodes to current route
            if len(current_route) > 0:
                routes.append(current_route)
            current_route = []
            current_load = 0.0

            # If still have unvisited nodes, start new route
            if unvisited:
                still_constructible = False
                for node in list(unvisited):
                    if wastes.get(node, 0.0) <= capacity:
                        still_constructible = True
                    else:
                        unvisited.remove(node)

                if not still_constructible:
                    break
                continue
            else:
                break

    # Add last route if not empty
    if len(current_route) > 0:
        routes.append(current_route)

    return routes
