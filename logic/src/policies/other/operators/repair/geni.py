"""
GENI (Generalized Insertion) Operator Module.

This module implements the exact GENI Type I and Type II constructive insertion
moves as defined by Gendreau, Hertz, and Laporte (1992). It inserts a node `u`
between two *non-adjacent* nodes in the tour while optimally reconnecting and
reversing the intermediate segments.

It contains:
1. `geni_insertion`: The standard, distance-minimizing operator for CVRP.
2. `geni_profit_insertion`: A profit-aware VRPP variant utilizing speculative
   seeding and economic pruning.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.geni import geni_insertion
    >>> improved = geni_insertion(ls, node=5, r_idx=0, neighborhood_size=5)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def prune_unprofitable_routes(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    mandatory_nodes_set: Set[int],
) -> List[List[int]]:
    """
    Evaluates all routes and removes those that result in a net economic loss,
    unless they contain mandatory nodes that must be served.
    """
    valid_routes = []
    for route in routes:
        if not route:
            continue

        if any(node in mandatory_nodes_set for node in route):
            valid_routes.append(route)
            continue

        cost = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i], route[i + 1]]
        cost += dist_matrix[route[-1], 0]

        revenue = sum(wastes.get(node, 0.0) for node in route) * R

        if (revenue - (cost * C)) >= -1e-4:
            valid_routes.append(route)

    return valid_routes


def _get_rev_cost(full_route: List[int], start: int, end: int, dist_matrix: np.ndarray) -> float:
    """Calculates the cost difference if the segment full_route[start:end] is reversed."""
    if start >= end - 1:
        return 0.0
    old_cost = sum(dist_matrix[full_route[k], full_route[k + 1]] for k in range(start, end - 1))
    new_cost = sum(dist_matrix[full_route[k + 1], full_route[k]] for k in range(start, end - 1))
    return new_cost - old_cost


def _apply_geni_move(route: List[int], u: int, i: int, j: int, m_type: str) -> List[int]:
    """Applies the exact structural reconnections for GENI insertions."""
    if m_type == "SIMPLE":
        return route[:i] + [u] + route[i:]

    full_route = [0] + route + [0]

    if m_type == "TYPE_I":
        # Deletes (v_i, v_{i+1}) & (v_j, v_{j+1}). Reverses v_{i+1}...v_j
        new_full = full_route[: i + 1] + [u] + full_route[i + 1 : j + 1][::-1] + full_route[j + 1 :]
    elif m_type == "TYPE_II":
        # Deletes (v_i, v_{i+1}) & (v_{j-1}, v_j). Reverses v_{i+1}...v_{j-1}
        new_full = full_route[: i + 1] + [u] + full_route[i + 1 : j][::-1] + full_route[j:]
    else:
        new_full = full_route

    return new_full[1:-1]  # Strip the depots back off


def _evaluate_route(
    u: int,
    route: List[int],
    dist_matrix: np.ndarray,
    neighborhood_size: int,
    revenue: Optional[float],
    C: float,
    is_man: bool,
) -> Tuple[float, Optional[Tuple[int, int, str]]]:
    """Evaluates all GENI moves for a specific route."""
    is_profit = revenue is not None
    best_val = -float("inf") if is_profit else float("inf")
    best_move = None

    full = [0] + route + [0]
    n_full = len(full)

    if neighborhood_size > 0 and (n_full - 1) > neighborhood_size:
        route_nodes = np.array(full[:-1])
        dists = dist_matrix[route_nodes, u]
        candidate_i = np.argsort(dists)[:neighborhood_size].tolist()
    else:
        candidate_i = list(range(n_full - 1))

    # Evaluate all move types
    for i in candidate_i:
        # 1. SIMPLE
        delta = dist_matrix[full[i], u] + dist_matrix[u, full[i + 1]] - dist_matrix[full[i], full[i + 1]]
        val = (revenue - delta * C) if is_profit else delta
        if (is_profit and val > best_val and (is_man or val >= -1e-4)) or (not is_profit and val < best_val):
            best_val, best_move = val, (i, -1, "SIMPLE")

        if n_full < 4:
            continue

        if i >= n_full - 2:
            continue

        for j in range(i + 2, n_full):
            # Type I
            if j < n_full - 1:
                rev_i = _get_rev_cost(full, i + 1, j + 1, dist_matrix)
                delta_i = (
                    dist_matrix[full[i], u]
                    + dist_matrix[u, full[j]]
                    + dist_matrix[full[i + 1], full[j + 1]]
                    - dist_matrix[full[i], full[i + 1]]
                    - dist_matrix[full[j], full[j + 1]]
                    + rev_i
                )
                val_i = (revenue - delta_i * C) if is_profit else delta_i
                if (is_profit and val_i > best_val and (is_man or val_i >= -1e-4)) or (
                    not is_profit and val_i < best_val
                ):
                    best_val, best_move = val_i, (i, j, "TYPE_I")

            # Type II
            rev_ii = _get_rev_cost(full, i + 1, j, dist_matrix)
            delta_ii = (
                dist_matrix[full[i], u]
                + dist_matrix[u, full[j - 1]]
                + dist_matrix[full[i + 1], full[j]]
                - dist_matrix[full[i], full[i + 1]]
                - dist_matrix[full[j - 1], full[j]]
                + rev_ii
            )
            val_ii = (revenue - delta_ii * C) if is_profit else delta_ii
            if (is_profit and val_ii > best_val and (is_man or val_ii >= -1e-4)) or (
                not is_profit and val_ii < best_val
            ):
                best_val, best_move = val_ii, (i, j, "TYPE_II")

    return best_val, best_move


def _find_best_geni_move(
    u: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    u_waste: float,
    capacity: float,
    neighborhood_size: int,
    revenue: Optional[float] = None,
    C: float = 1.0,
    mandatory_set: Optional[Set[int]] = None,
) -> Tuple[float, Optional[Tuple[int, int, int, str]]]:
    """Finds best GENI move for node u across all routes."""
    is_profit = revenue is not None
    best_val = -float("inf") if is_profit else float("inf")
    best_move = None
    is_man = u in (mandatory_set or set())

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + u_waste > capacity:
            continue

        val, move = _evaluate_route(u, route, dist_matrix, neighborhood_size, revenue, C, is_man)
        if move and ((is_profit and val > best_val) or (not is_profit and val < best_val)):
            best_val = val
            best_move = (r_idx, move[0], move[1], move[2])

    return best_val, best_move


def geni_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    neighborhood_size: int = 5,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Standard GENI insertion. Inserts nodes to strictly minimize total distance
    using Simple, Type I, and Type II topology bypasses.

    Args:
        routes: List of active routes.
        removed_nodes: Nodes needing re-insertion.
        dist_matrix: Network distance matrix.
        wastes: Dictionary of node demands.
        capacity: Max load per vehicle.
        neighborhood_size: Restricts the search for v_i to the k-nearest nodes to u in the route.
        mandatory_nodes: List of mandatory nodes ensuring safety fallback insertions.
        rng: Random number generator.
        expand_pool: Whether to expand the pool of candidate nodes.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    if rng is None:
        rng = Random(42)

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    rng.shuffle(unassigned)
    for u in unassigned:
        u_waste = wastes.get(u, 0.0)
        best_delta, best_move = _find_best_geni_move(
            u, routes, loads, dist_matrix, u_waste, capacity, neighborhood_size, mandatory_set=mandatory_set
        )

        new_cost = dist_matrix[0, u] + dist_matrix[u, 0]
        if new_cost < best_delta:
            best_delta, best_move = new_cost, (len(routes), 0, 0, "NEW")

        if best_move:
            r_idx, i, j, m_type = best_move
            if m_type == "NEW":
                routes.append([u])
                loads.append(u_waste)
            else:
                routes[r_idx] = _apply_geni_move(routes[r_idx], u, i, j, m_type)
                loads[r_idx] += u_waste
        elif u in mandatory_set:
            routes.append([u])
            loads.append(u_waste)

    return routes


def geni_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_size: int = 5,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Profit-aware VRPP GENI insertion with speculative seeding."""
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    if rng is None:
        rng = Random(42)

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    rng.shuffle(unassigned)
    for u in unassigned:
        u_waste = wastes.get(u, 0.0)
        revenue = u_waste * R
        best_profit, best_move = _find_best_geni_move(
            u, routes, loads, dist_matrix, u_waste, capacity, neighborhood_size, revenue, C, mandatory_set
        )

        new_cost_c = (dist_matrix[0, u] + dist_matrix[u, 0]) * C
        new_profit = revenue - new_cost_c
        seed_hurdle = -0.5 * new_cost_c

        if new_profit > best_profit and (u in mandatory_set or new_profit >= seed_hurdle):
            best_profit, best_move = new_profit, (len(routes), 0, 0, "NEW")

        if best_move:
            r_idx, i, j, m_type = best_move
            if m_type == "NEW":
                routes.append([u])
                loads.append(u_waste)
            else:
                routes[r_idx] = _apply_geni_move(routes[r_idx], u, i, j, m_type)
                loads[r_idx] += u_waste
        elif u in mandatory_set:
            routes.append([u])
            loads.append(u_waste)

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
