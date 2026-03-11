"""
Sector Removal Operator Module.

Calculates the angle of each customer relative to the depot, picks a random
starting angle, and sweeps a sector removing all nodes within that angular
region until ``k`` nodes have been removed.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.sector import sector_removal
    >>> routes, removed = sector_removal(routes, n_remove=5, coords=coords, depot=(0.5, 0.5))
"""

import math
from random import Random
from typing import List, Optional, Tuple

import numpy as np


def sector_removal(
    routes: List[List[int]],
    n_remove: int,
    coords: np.ndarray,
    depot: Tuple[float, float] = (0.0, 0.0),
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes within an angular sector from the depot.

    Converts customer coordinates to polar angles centred on the depot,
    picks a random starting angle, and sweeps until ``n_remove`` nodes
    are collected.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        coords: Node coordinates, shape ``(N+1, 2)``.  Index 0 = depot.
        depot: Depot coordinates ``(x, y)`` (default uses ``coords[0]``).
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random(42)

    # Gather nodes and compute angles
    node_angles: List[Tuple[float, int, int, int]] = []  # (angle, node, r_idx, pos)
    depot_x, depot_y = depot

    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            if node < len(coords):
                dx = float(coords[node, 0]) - depot_x
                dy = float(coords[node, 1]) - depot_y
                angle = math.atan2(dy, dx)  # range [-pi, pi]
            else:
                angle = 0.0
            node_angles.append((angle, node, r_idx, pos))

    if not node_angles:
        return routes, []

    # Sort by angle
    node_angles.sort(key=lambda x: x[0])
    n_total = len(node_angles)
    n_remove = min(n_remove, n_total)

    # Pick random starting angle
    start_angle = rng.uniform(-math.pi, math.pi)

    # Find the starting index (first node at or after start_angle)
    start_idx = 0
    for i, (angle, _, _, _) in enumerate(node_angles):
        if angle >= start_angle:
            start_idx = i
            break

    # Sweep from start_idx, collecting n_remove nodes
    to_remove: List[Tuple[int, int, int]] = []  # (node, r_idx, pos)
    for offset in range(n_total):
        if len(to_remove) >= n_remove:
            break
        idx = (start_idx + offset) % n_total
        _, node, r_idx, pos = node_angles[idx]
        to_remove.append((node, r_idx, pos))

    # Sort by (route_idx, position) descending for safe removal
    to_remove.sort(key=lambda x: (x[1], x[2]), reverse=True)

    removed: List[int] = []
    for node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
