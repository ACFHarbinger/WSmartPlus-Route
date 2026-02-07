"""
Destroy operators for the Adaptive Large Neighborhood Search (ALNS).

This module contains various removal heuristics used to discard nodes
from the current routing solution to explore the search space.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def cluster_removal(
    routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray, nodes: List[int]
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a cluster of nodes based on spatial proximity (Shaw Removal variant).

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        nodes (List[int]): List of all node IDs.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Pick a random node, remove it and its k nearest neighbors
    if not any(routes):
        return routes, []

    # Pick seed
    seed_route_idx = random.randint(0, len(routes) - 1)
    if not routes[seed_route_idx]:
        return random_removal(routes, n_remove)

    seed_node = random.choice(routes[seed_route_idx])

    removed = [seed_node]

    # Get all nodes current pos
    node_map = {}
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            node_map[node] = (r_idx, n_idx)

    # Find neighbors
    candidates = []
    for v in nodes:
        if v == seed_node or v not in node_map:
            continue
        dist = dist_matrix[seed_node][v]
        candidates.append((v, dist))

    candidates.sort(key=lambda x: x[1])

    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed.extend(target_nodes)

    # Now remove them from routes
    to_remove_locs = []
    for node in removed:
        if node in node_map:
            to_remove_locs.append((*node_map[node], node))

    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, n_idx, node in to_remove_locs:
        routes[r_idx].pop(n_idx)
        final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed


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


def string_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    max_string_len: int = 5,
    avg_string_len: float = 3.0,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers to induce spatial slack.

    The key insight of SISR is that removing adjacent customers creates a large
    contiguous "hole" in the route, providing maneuverability for reinsertion.

    Args:
        routes: Current routes.
        n_remove: Target number of nodes to remove.
        dist_matrix: Distance matrix for neighbor lookup.
        max_string_len: Maximum length of a single string.
        avg_string_len: Average string length (controls distribution).

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    removed: List[int] = []
    max_iter = n_remove * 3  # Prevent infinite loops
    iterations = 0

    while len(removed) < n_remove and iterations < max_iter:
        iterations += 1

        # Pick a random seed from remaining nodes
        available_routes = [(i, r) for i, r in enumerate(routes) if r]
        if not available_routes:
            break

        r_idx, route = random.choice(available_routes)
        if not route:
            continue

        # Pick seed position
        seed_pos = random.randint(0, len(route) - 1)
        seed_node = route[seed_pos]

        if seed_node in removed:
            continue

        # Determine string length (geometric-like distribution)
        # L ~ 1 + geometric(1/avg_string_len)
        string_len = 1
        while string_len < max_string_len and random.random() < (1 - 1 / avg_string_len):
            string_len += 1

        # Don't remove more than needed
        remaining = n_remove - len(removed)
        string_len = min(string_len, remaining, len(route))

        # Extract string starting at seed_pos
        start = seed_pos
        end = min(seed_pos + string_len, len(route))
        string_nodes = route[start:end]

        # Remove string from route (reverse order to maintain indices)
        for pos in range(end - 1, start - 1, -1):
            node = routes[r_idx].pop(pos)
            removed.append(node)

        # Propagate to neighbors: remove strings from adjacent routes
        if len(removed) < n_remove and string_nodes:
            _propagate_string_removal(routes, removed, dist_matrix, string_nodes, n_remove, max_string_len)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed


def _propagate_string_removal(
    routes: List[List[int]],
    removed: List[int],
    dist_matrix: np.ndarray,
    seed_nodes: List[int],
    n_remove: int,
    max_string_len: int,
) -> None:
    """
    Propagate string removal to neighboring routes.

    After removing a string, look at the spatial neighbors of the removed nodes
    and remove strings from their routes as well. This creates a concentrated
    "disaster zone" across multiple routes.
    """
    # Find neighbors of removed string
    neighbor_candidates = []
    for seed in seed_nodes:
        if seed >= len(dist_matrix):
            continue
        distances = dist_matrix[seed]
        for node_id in range(1, len(distances)):
            if node_id not in removed and node_id not in seed_nodes:
                neighbor_candidates.append((node_id, distances[node_id]))

    # Sort by distance
    neighbor_candidates.sort(key=lambda x: x[1])

    # Take closest neighbors
    for neighbor, _ in neighbor_candidates[:3]:
        if len(removed) >= n_remove:
            break

        # Find which route contains this neighbor
        for r_idx, route in enumerate(routes):
            if neighbor in route:
                pos = route.index(neighbor)
                # Remove a small string around this neighbor
                string_len = min(2, len(route), n_remove - len(removed))
                start = max(0, pos - string_len // 2)
                end = min(len(route), start + string_len)

                for p in range(end - 1, start - 1, -1):
                    if routes[r_idx][p] not in removed:
                        removed.append(routes[r_idx].pop(p))
                break
