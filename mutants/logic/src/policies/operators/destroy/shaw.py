import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def shaw_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    demands: Optional[Dict[int, float]] = None,
    time_windows: Optional[Dict[int, tuple]] = None,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    randomization_factor: float = 2.0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Shaw Removal: Remove related customers based on multi-criteria similarity.

    Relatedness R(i,j) = phi * d(i,j) + chi * |T_i - T_j| + psi * |q_i - q_j|

    Customers that are "similar" (close in space, time, and demand) are removed
    together, maximizing the potential for rearrangement during repair.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        demands: Node demand dictionary {node: demand}.
        time_windows: Time window dict {node: (earliest, latest)}.
        phi: Distance weight in relatedness.
        chi: Time window weight in relatedness.
        psi: Demand weight in relatedness.
        randomization_factor: Power for randomized selection (higher = more random).

    Returns:
        Tuple of (modified routes, removed nodes).
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    demands = demands or {}
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

    # Pick random seed
    seed: int = random.choice(all_nodes)
    removed: List[int] = [seed]

    # Normalize distance for relatedness calculation
    max_dist: float = float(np.max(dist_matrix)) if np.max(dist_matrix) > 0 else 1.0
    max_demand: float = float(max(demands.values())) if demands else 1.0
    max_tw: float = 1.0
    if time_windows:
        tw_spans = [tw[1] - tw[0] for tw in time_windows.values() if tw]
        max_tw = float(max(tw_spans)) if tw_spans else 1.0

    while len(removed) < n_remove and len(removed) < len(all_nodes):
        # Calculate relatedness to already-removed nodes
        relatedness_scores: List[Tuple[int, float]] = []

        for node in all_nodes:
            if node in removed or node not in node_map:
                continue

            # Average relatedness to all removed nodes
            total_rel: float = 0.0
            for rem_node in removed:
                # Distance component
                dist_rel: float = float(dist_matrix[node, rem_node]) / max_dist if max_dist > 0 else 0.0

                # Demand component
                dem_rel: float = 0.0
                if demands:
                    dem_rel = float(abs(demands.get(node, 0.0) - demands.get(rem_node, 0.0))) / max_demand

                # Time window component
                tw_rel: float = 0.0
                if time_windows:
                    tw_node = time_windows.get(node, (0.0, max_tw))
                    tw_rem = time_windows.get(rem_node, (0.0, max_tw))
                    tw_rel = float(abs(tw_node[0] - tw_rem[0])) / max_tw

                total_rel += float(phi * dist_rel + chi * tw_rel + psi * dem_rel)

            avg_rel: float = total_rel / len(removed)
            relatedness_scores.append((node, avg_rel))

        if not relatedness_scores:
            break

        # Sort by relatedness (lower = more related)
        relatedness_scores.sort(key=lambda x: x[1])

        # Randomized selection using power law
        # y^p where y is uniform [0,1], p is randomization_factor
        y = random.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        idx = min(idx, len(relatedness_scores) - 1)
        selected_node = relatedness_scores[idx][0]
        removed.append(selected_node)

    # Remove from routes
    to_remove_locs = [(node_map[n][0], node_map[n][1], n) for n in removed if n in node_map]
    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, pos, node in to_remove_locs:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed
