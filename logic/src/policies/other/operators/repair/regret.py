"""
Regret Insertion Operator Module.

This module implements the Regret-k insertion heuristic for VRP repair.
It calculates a regret value for each unassigned node based on the cost difference
between its best and k-th best insertion positions.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.repair.regret import RegretInsertion
    >>> operator = RegretInsertion(k=2)
    >>> new_routes = operator.repair(destroyed_routes, unassigned_nodes)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ._prune_routes import prune_unprofitable_routes


def regret_2_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes based on the regret-2 criterion.
    Regret-2 Insertion Heuristic.

    Prioritizes inserting nodes that would be much more expensive to insert
    later (high regret = cost difference between best and 2nd best option).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: Optional list of mandatory node indices.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes using the regret-k heuristic.

    Computes the 'regret' for each unassigned node, defined as the difference
    between the cost of the best insertion and the k-th best insertion. Node
    with the maximum regret is inserted first into its best position.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        k: Regret degree (2, 3, etc.).
        mandatory_nodes: Optional list of mandatory node indices.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    # Calculate current loads and track visited
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        all_candidates = []
        unprofitable_nodes = []

        for node in unassigned:
            node_waste = wastes.get(node, 0)
            is_mandatory = node in mandatory_nodes_set

            node_options = []
            # Check all existing routes
            for i, route in enumerate(routes):
                if loads[i] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    # Calculate base insertion cost
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                    # Apply noise additively (Ropke & Pisinger 2005, Section 3.4.3)
                    # C' = max{0, C + noise} where noise is a perturbation
                    if noise != 0:
                        cost = max(0.0, cost + noise)

                    node_options.append((cost, i, pos))

            # Also consider starting a new route if distance allows (not always applicable, but good for completeness)
            # In ALNS, we usually only work with existing routes provided or let constructive do it.

            if not node_options:
                if not is_mandatory:
                    unprofitable_nodes.append(node)
                continue

            # Sort candidate positions for this node by cost
            node_options.sort(key=lambda x: x[0])

            # Calculate regret
            if len(node_options) >= k:
                regret = node_options[k - 1][0] - node_options[0][0]
            elif len(node_options) > 1:
                # If fewer than k options, regret is diff between last and first
                regret = node_options[-1][0] - node_options[0][0]
            else:
                # Only one option (or none), max priority
                regret = float("inf")

            best_option = node_options[0] if node_options else (float("inf"), -1, -1)
            all_candidates.append((regret, node, best_option))

        # Remove unprofitable nodes from unassigned
        for node in unprofitable_nodes:
            unassigned.remove(node)

        if not all_candidates:
            # No feasible/profitable insertions left
            # For remaining mandatory nodes, we'll hit this break and need to handle them
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                continue
            else:
                break

        # Pick node with max regret, tie-break by node ID
        all_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, best_node, (cost, r_idx, pos) = all_candidates[0]

        if r_idx == -1:
            # Cannot insert node anywhere
            unassigned.remove(best_node)
            continue

        # Apply insertion
        waste = wastes.get(best_node, 0)
        if r_idx == len(routes):
            routes.append([best_node])
            loads.append(waste)
        else:
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += waste

        unassigned.remove(best_node)

    return routes


def _get_insertion_options_with_profit(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    is_mandatory: bool,
    noise: float,
    max_dist: float,
) -> List[Tuple[float, int, int]]:
    """Helper to calculate insertion options for a node with profit logic."""
    node_waste = wastes.get(node, 0)
    revenue = node_waste * R
    node_options = []

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

            if noise != 0:
                cost += noise * max_dist

            profit = revenue - cost * C

            # Requirement: profit > 0 or is_mandatory
            if is_mandatory or profit > -1e-4:
                node_options.append((profit, i, pos))

    # Evaluate new route (Speculative Seeding Heuristic)
    # See greedy_profit_insertion for theoretical justification
    new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
    new_profit = revenue - (new_cost * C)
    seed_hurdle = -0.5 * (new_cost * C)  # Speculative hurdle: 50% of detour cost

    if is_mandatory or new_profit >= seed_hurdle:
        # A new route has its best (and only) insertion at pos 0 in a new index
        node_options.append((new_profit, len(routes), 0))

    return node_options


def regret_2_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes based on the regret-2 criterion maximizing profit.

    Wrapper around regret_k_profit_insertion with k=2.
    """
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Regret-k insertion maximizing profit (revenue - cost).

    VRPP logic: Instead of minimizing cost, we calculate profit for each position.
    A node is only considered if its best insertion is profitable or if it's mandatory.
    Regret is calculated as the difference between the best and k-th best profits.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        k: Regret degree.
        mandatory_nodes: List of mandatory node indices.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0) for node in r) for r in routes]

    visited = set()
    for r in routes:
        visited.update(r)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    max_dist = dist_matrix.max()

    while unassigned:
        all_candidates = []
        skipped_nodes = []

        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            node_options = _get_insertion_options_with_profit(
                node,
                routes,
                loads,
                dist_matrix,
                wastes,
                capacity,
                R,
                C,
                is_mandatory,
                noise,
                max_dist,
            )

            if not node_options:
                if not is_mandatory:
                    skipped_nodes.append(node)
                continue

            # Sort by profit (highest first)
            node_options.sort(key=lambda x: x[0], reverse=True)

            # Regret for profit: best_profit - k_th_best_profit
            if len(node_options) >= k:
                regret = node_options[0][0] - node_options[k - 1][0]
            elif len(node_options) > 1:
                regret = node_options[0][0] - node_options[-1][0]
            else:
                regret = float("inf")

            best_option = node_options[0]
            all_candidates.append((regret, node, best_option))

        for out_node in skipped_nodes:
            unassigned.remove(out_node)

        if not all_candidates:
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                continue
            else:
                break
        # Maximize regret
        all_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, best_node, (profit, r_idx, pos) = all_candidates[0]

        if r_idx == -1:
            unassigned.remove(best_node)
            continue

        waste = wastes.get(best_node, 0)
        if r_idx == len(routes):
            routes.append([best_node])
            loads.append(waste)
        else:
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += waste

        unassigned.remove(best_node)

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
