"""
Sector Removal Operator Module.

Calculates the angle of each customer relative to the depot, picks a random
starting angle, and sweeps a sector removing all nodes within that angular
region until ``k`` nodes have been removed.

Also includes profit-based sector removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.sector import sector_removal
    >>> routes, removed = sector_removal(routes, n_remove=5, coords=coords, depot=(0.5, 0.5))
    >>> from logic.src.policies.other.operators.destroy.sector import sector_profit_removal
    >>> routes, removed = sector_profit_removal(routes, n_remove=5, coords=coords, dist_matrix=d,
    ...                                          wastes=w, R=1.0, C=1.0, depot=(0.5, 0.5))
"""

import math
from random import Random
from typing import Dict, List, Optional, Tuple

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


def _calculate_node_profits(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> Dict[int, float]:
    """Calculate the profit (revenue - cost) for each node currently in the routes."""
    node_profits: Dict[int, float] = {}
    for route in routes:
        for node in route:
            if node not in node_profits:
                revenue = wastes.get(node, 0.0) * R
                cost = dist_matrix[0][node] * C
                node_profits[node] = revenue - cost
    return node_profits


def _choose_starting_angle(
    node_angles: List[Tuple[float, int, int, int, float]],
    bias_low_profit: bool,
    rng: Random,
) -> float:
    """Choose a starting angle, optionally biased toward sectors with low average profit."""
    n_total = len(node_angles)
    if not bias_low_profit or not node_angles:
        # Random starting angle (standard behavior)
        return rng.uniform(-math.pi, math.pi)

    # Bias toward low-profit sectors
    # Divide circle into sectors and find sector with lowest average profit
    n_sectors = min(8, n_total)  # Use 8 sectors or fewer if not enough nodes
    sector_size = n_total / n_sectors
    sector_profits = []

    for i in range(n_sectors):
        start_idx = int(i * sector_size)
        end_idx = int((i + 1) * sector_size)
        sector_nodes = node_angles[start_idx:end_idx]
        avg_profit = sum(p for _, _, _, _, p in sector_nodes) / max(len(sector_nodes), 1)
        sector_profits.append((i, avg_profit, sector_nodes[0][0] if sector_nodes else 0.0))

    # Sort sectors by profit (ascending - worst first)
    sector_profits.sort(key=lambda x: x[1])

    # Select from bottom 25% of sectors with some randomization
    bottom_quartile = max(1, len(sector_profits) // 4)
    low_profit_sectors = sector_profits[:bottom_quartile]
    _, _, start_angle = rng.choice(low_profit_sectors)

    return start_angle


def sector_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    coords: np.ndarray,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    depot: Tuple[float, float] = (0.0, 0.0),
    bias_low_profit: bool = True,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes within an angular sector, biased toward low-profit regions.

    Converts customer coordinates to polar angles centred on the depot, then
    selects a starting angle biased toward low-profit nodes. Sweeps from that
    angle until ``n_remove`` nodes are collected.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        coords: Node coordinates, shape ``(N+1, 2)``. Index 0 = depot.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        depot: Depot coordinates ``(x, y)`` (default uses ``coords[0]``).
        bias_low_profit: If True, bias starting angle toward low-profit sectors.
                        If False, use random starting angle (standard sector removal).
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random(42)

    # Calculate profit for all nodes
    node_profits = _calculate_node_profits(routes, dist_matrix, wastes, R, C)

    # Gather nodes and compute angles
    node_angles: List[Tuple[float, int, int, int, float]] = []  # (angle, node, r_idx, pos, profit)
    depot_x, depot_y = depot

    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            if node < len(coords):
                dx = float(coords[node, 0]) - depot_x
                dy = float(coords[node, 1]) - depot_y
                angle = math.atan2(dy, dx)  # range [-pi, pi]
            else:
                angle = 0.0
            profit = node_profits.get(node, 0.0)
            node_angles.append((angle, node, r_idx, pos, profit))

    if not node_angles:
        return routes, []

    # Sort by angle
    node_angles.sort(key=lambda x: x[0])
    n_total = len(node_angles)
    n_remove = min(n_remove, n_total)

    # Choose starting angle
    start_angle = _choose_starting_angle(node_angles, bias_low_profit, rng)

    # Find the starting index (first node at or after start_angle)
    start_idx = 0
    for i, (angle, _, _, _, _) in enumerate(node_angles):
        if angle >= start_angle:
            start_idx = i
            break

    # Sweep from start_idx, collecting n_remove nodes
    to_remove: List[Tuple[int, int, int]] = []  # (node, r_idx, pos)
    for offset in range(n_total):
        if len(to_remove) >= n_remove:
            break
        idx = (start_idx + offset) % n_total
        _, node, r_idx, pos, _ = node_angles[idx]
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
