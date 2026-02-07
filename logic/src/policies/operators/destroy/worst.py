from typing import List, Tuple

import numpy as np


def worst_removal(routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes that contribute most to the current routing cost.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Remove nodes that contribute most to the cost
    costs = []
    for r_idx, route in enumerate(routes):
        if len(route) == 0:
            continue

        for i, node in enumerate(route):
            # Calc cost without this node
            # Prev -> Next
            prev = 0 if i == 0 else route[i - 1]
            nex = 0 if i == len(route) - 1 else route[i + 1]

            # We save dist(prev, node) + dist(node, nex)
            saved = dist_matrix[prev][node] + dist_matrix[node][nex]
            # We add dist(prev, nex)
            added = dist_matrix[prev][nex]

            # Savings = saved - added
            savings = saved - added
            costs.append((r_idx, i, node, savings))

    costs.sort(key=lambda x: x[3], reverse=True)  # Highest savings first
    removed = []

    # One-shot greedy:
    targets = costs[:n_remove]
    # Sort by index desc
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node, _ in targets:
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx].pop(n_idx)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
