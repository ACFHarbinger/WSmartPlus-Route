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
from typing import Any, Dict, List, Optional, Set, Tuple

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

    Efficiency: O(n) selection using np.partition instead of O(n log n) sorting.
    Implements Ropke & Pisinger (2005) relatedness measure:
        R(i,j) = φ * d(i,j) + χ * |T_i - T_j| + ψ * |q_i - q_j|

    **VRPP Adaptation Note**: While the Ropke & Pisinger (2005) measure includes
    temporal components (T_ij), the VRPP-specific relatedness primarily relies
    on spatial distance and profit difference. This implementation drops the
    vehicle compatibility term (ω) and emphasizes profit attributes.

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is intentionally
    omitted from this implementation. Theoretical justification: The target domain is
    CVRP/VRPP with a homogeneous fleet.

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

        # Structured array for O(n) partitioning with deterministic tie-breaking (score, then node ID)
        dtype = [("node", "i4"), ("score", "f4")]
        arr = np.array(relatedness_scores, dtype=dtype)

        # Randomized selection using power law
        # y^p where y is uniform [0,1], p is randomization_factor
        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        k = min(idx, len(relatedness_scores) - 1)

        # O(n) selection of the k-th smallest element
        # np.partition ensures elements at k are in their sorted position
        partitioned = np.partition(arr, k, order=["score", "node"])
        selected_node = int(partitioned[k]["node"])
        removed.append(selected_node)

    # Execution removals: efficient route modification using list filtering
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = [node for node in r if node not in removed_set]
        # Track removals from this specific route for final_removed
        for node in r:
            if node in removed_set:
                final_removed.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed


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

    Efficiency: O(n) selection using np.partition instead of O(n log n) sorting.
    Implements profit-aware relatedness for VRPP problems:
        R(i,j) = φ * d(i,j)/d_max + ψ * |profit_i - profit_j|/profit_max

    Calculates relatedness purely based on spatial distance and profit difference,
    dropping the temporal components (T_ij) of standard PDPTW Shaw Removal.

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is intentionally
    omitted as the fleet is homogeneous.

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
                dist_rel = float(dist_matrix[node, rem_node]) / max_dist if max_dist > 0 else 0.0
                profit_rel = float(abs(node_profits[node] - node_profits[rem_node])) / profit_range

                rel = phi * dist_rel + psi * profit_rel
                total_rel += rel

            avg_rel = total_rel / len(removed)
            relatedness_scores.append((node, avg_rel))

        if not relatedness_scores:
            break

        # Structured array for O(n) partitioning with deterministic tie-breaking (score, then node ID)
        dtype = [("node", "i4"), ("score", "f4")]
        arr = np.array(relatedness_scores, dtype=dtype)

        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        k = min(idx, len(relatedness_scores) - 1)

        # O(n) selection of the k-th smallest element
        partitioned = np.partition(arr, k, order=["score", "node"])
        selected_node = int(partitioned[k]["node"])
        removed.append(selected_node)

    # 4. Efficient route modification
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = [node for node in r if node not in removed_set]
        # Track removals from this specific route for final_removed
        for node in r:
            if node in removed_set:
                final_removed.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed
