"""
Shaw Removal (Relatedness) Operator Module.

This module implements the Shaw removal heuristic, which removes nodes that are
similar (related) to a seed node based on distance, time, and waste.

Also includes profit-based Shaw removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.destroy.shaw import shaw_removal
    >>> routes, removed = shaw_removal(routes, n_remove=5, dist_matrix=d, ...)
    >>> from logic.src.policies.other.operators.destroy.shaw import shaw_profit_removal
    >>> routes, removed = shaw_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def shaw_removal(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Optional[Dict[int, float]] = None,
    time_windows: Optional[Dict[Any, Any]] = None,
    randomization_factor: float = 2.0,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Shaw Removal: Remove related customers based on multi-criteria similarity.

    Implements Ropke & Pisinger (2005) relatedness measure:
        R(i,j) = φ * d(i,j) + χ * |T_i - T_j| + ψ * |q_i - q_j| + ω * vehicle_compatibility

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is intentionally
    omitted from this implementation. Theoretical justification: The target domain is
    CVRP/VRPP with a homogeneous fleet where any vehicle can serve any node. Since all
    nodes have identical vehicle compatibility, the ω term is mathematically constant
    across all node pairs and does not affect relative node selection. Therefore, it can
    be safely omitted without changing the algorithm's behavior.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        time_windows: Time window dict {node: (earliest, latest)} (optional).
        randomization_factor: Power for randomized selection (higher = more random).
        phi: Distance weight in relatedness (φ).
        chi: Time window weight in relatedness (χ).
        psi: Waste/demand weight in relatedness (ψ).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed nodes.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    waste = wastes or {}
    time_windows = time_windows or {}

    # Build node map
    node_map = {}
    all_nodes = []
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_map[node] = (r_idx, pos)
            all_nodes.append(node)

    if not all_nodes:
        return routes, []

    if rng is None:
        rng = Random(42)

    # Pick random seed
    seed: int = rng.choice(all_nodes)
    removed: List[int] = [seed]

    # Normalize factors using only active route nodes (Ropke & Pisinger 2005)
    # This prevents geographic outliers from compressing all distance scores to near-zero
    # Calculate max distance only among nodes currently in routes
    active_distances = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                active_distances.append(dist_matrix[i, j])
    max_dist: float = float(max(active_distances)) if active_distances else 1.0

    max_waste: float = float(max(waste.values())) if waste else 1.0
    max_tw: float = 1.0
    if time_windows:
        tw_spans = [tw[1] - tw[0] for tw in time_windows.values() if tw]
        max_tw = float(max(tw_spans)) if tw_spans else 1.0

    while len(removed) < n_remove and len(removed) < len(all_nodes):
        relatedness_scores: List[Tuple[int, float]] = []

        for node in all_nodes:
            if node in removed or node not in node_map:
                continue

            total_rel: float = 0.0
            for rem_node in removed:
                dist_rel = float(dist_matrix[node, rem_node]) / max_dist if max_dist > 0 else 0.0
                waste_rel = 0.0
                if waste:
                    waste_rel = float(abs(waste.get(node, 0.0) - waste.get(rem_node, 0.0))) / max_waste
                tw_rel = 0.0
                if time_windows:
                    tw_node = time_windows.get(node, (0.0, max_tw))
                    tw_rem = time_windows.get(rem_node, (0.0, max_tw))
                    tw_rel = float(abs(tw_node[0] - tw_rem[0])) / max_tw

                rel = phi * dist_rel + chi * tw_rel + psi * waste_rel
                total_rel += rel

            avg_rel = total_rel / len(removed)
            relatedness_scores.append((node, avg_rel))

        if not relatedness_scores:
            break

        # Sort by relatedness (lower = more related), then node ID for deterministic tie-breaking
        relatedness_scores.sort(key=lambda x: (x[1], x[0]))

        # Randomized selection using power law
        # y^p where y is uniform [0,1], p is randomization_factor
        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        selected_node = relatedness_scores[min(idx, len(relatedness_scores) - 1)][0]
        removed.append(selected_node)

    # Execution removals
    to_remove_locs = [(node_map[n][0], node_map[n][1], n) for n in removed if n in node_map]
    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, pos, node in to_remove_locs:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed


def shaw_profit_removal(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    randomization_factor: float = 2.0,
    phi: float = 9.0,
    psi: float = 5.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Profit-based Shaw Removal (VRPP).

    Implements profit-aware relatedness for VRPP problems:
        R(i,j) = φ * d(i,j)/d_max + ψ * |profit_i - profit_j|/profit_max

    Customers that are similar in terms of distance and marginal profit are removed
    together, maximizing the potential for profitable rearrangement during repair.

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is intentionally
    omitted from this implementation. Theoretical justification: The target domain is
    VRPP with a homogeneous fleet where any vehicle can serve any node. Since all nodes
    have identical vehicle compatibility, the ω term is mathematically constant across
    all node pairs and does not affect relative node selection.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        randomization_factor: Power for randomized selection (higher = more random).
        phi: Distance weight in relatedness (φ).
        psi: Profit weight in relatedness (ψ).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed nodes.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    # 1. Pre-calculate node profits using marginal formula
    node_map = {}
    all_nodes = []
    node_profits = {}
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_map[node] = (r_idx, pos)
            all_nodes.append(node)

            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            node_profits[node] = revenue - (detour_cost * C)

    if not all_nodes:
        return routes, []

    if rng is None:
        rng = Random(42)

    # 2. Pick random seed
    seed: int = rng.choice(all_nodes)
    removed: List[int] = [seed]

    # 3. Normalization factors using only active route nodes
    # Calculate max distance only among nodes currently in routes
    active_distances = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                active_distances.append(dist_matrix[i, j])
    max_dist = float(max(active_distances)) if active_distances else 1.0

    profit_vals = list(node_profits.values())
    profit_range = float(max(profit_vals) - min(profit_vals)) if len(profit_vals) > 1 else 1.0
    if profit_range == 0:
        profit_range = 1.0

    while len(removed) < n_remove and len(removed) < len(all_nodes):
        relatedness_scores: List[Tuple[int, float]] = []

        for node in all_nodes:
            if node in removed or node not in node_map:
                continue

            total_rel: float = 0.0
            for rem_node in removed:
                dist_rel = float(dist_matrix[node, rem_node]) / max_dist
                profit_rel = float(abs(node_profits[node] - node_profits[rem_node])) / profit_range

                rel = phi * dist_rel + psi * profit_rel
                total_rel += rel

            avg_rel = total_rel / len(removed)
            relatedness_scores.append((node, avg_rel))

        if not relatedness_scores:
            break

        relatedness_scores.sort(key=lambda x: (x[1], x[0]))
        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        selected_node = relatedness_scores[min(idx, len(relatedness_scores) - 1)][0]
        removed.append(selected_node)

    # 4. Final removal
    to_remove_locs = [(node_map[n][0], node_map[n][1], n) for n in removed if n in node_map]
    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, pos, node in to_remove_locs:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed
