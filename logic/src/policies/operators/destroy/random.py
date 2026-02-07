import random
from typing import List, Tuple


def random_removal(routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
    """
    Randomly remove n_remove nodes from the current routes.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    removed = []
    # Flatten
    all_nodes = []
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            all_nodes.append((r_idx, n_idx, node))

    if n_remove >= len(all_nodes):
        return [[]], [n for _, _, n in all_nodes]

    targets = random.sample(all_nodes, n_remove)

    # Sort targets by r_idx, n_idx desc to pop safely
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node in targets:
        routes[r_idx].pop(n_idx)
        removed.append(node)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed
