"""
Heuristic Algorithms for VRPP Primal Solutions.

This module provides construction heuristics for finding good feasible VRPP solutions
to provide upper bounds for the branch-and-cut algorithm. It leverages existing
operators from the WSmart+ Route codebase.

Based on Section 4 of Fischetti et al. (1997) and classical VRP heuristics.
"""

from random import Random
from typing import List, Tuple

from logic.src.policies.branch_and_cut.vrpp_model import VRPPModel
from logic.src.policies.other.operators.heuristics.greedy_initialization import build_greedy_routes
from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes
from logic.src.policies.other.operators.repair.farthest import (
    farthest_insertion as operator_farthest_insertion,
)
from logic.src.policies.other.operators.repair.farthest import (
    farthest_profit_insertion as operator_farthest_profit_insertion,
)


def construct_initial_solution(model: VRPPModel, seed: int = 42) -> Tuple[List[int], float]:
    """
    Construct an initial feasible VRPP solution using greedy initialization.

    This function uses the build_greedy_routes operator to build a solution that
    maximizes profit (revenue - cost) while respecting capacity constraints.

    Args:
        model: VRPPModel instance.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (tour, profit).
    """
    rng = Random(seed)

    # Use greedy initialization to build routes
    routes = build_greedy_routes(
        dist_matrix=model.cost_matrix,
        wastes=model.wastes,
        capacity=model.capacity,
        R=model.R,
        C=model.C,
        mandatory_nodes=list(model.mandatory_nodes) if model.mandatory_nodes else None,
        rng=rng,
    )

    # Convert routes to a single tour
    tour = _routes_to_tour(routes, model.depot)

    # Improve with 2-opt
    tour = _apply_2opt_to_tour(model, tour)

    # Calculate profit
    profit = model.compute_tour_profit(tour)

    return tour, profit


def construct_nn_solution(model: VRPPModel, seed: int = 42) -> Tuple[List[int], float]:
    """
    Construct an initial solution using Nearest Neighbor initialization.

    This uses the build_nn_routes function to create geographically compact routes.

    Args:
        model: VRPPModel instance.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (tour, profit).
    """
    rng = Random(seed)

    # Get all customer nodes
    all_customers = list(range(1, model.n_nodes))

    # Build NN routes
    routes = build_nn_routes(
        nodes=all_customers,
        mandatory_nodes=list(model.mandatory_nodes) if model.mandatory_nodes else [],
        wastes=model.wastes,
        capacity=model.capacity,
        dist_matrix=model.cost_matrix,
        R=model.R,
        C=model.C,
        rng=rng,
    )

    # Convert to single tour
    tour = _routes_to_tour(routes, model.depot)

    # Improve with 2-opt
    tour = _apply_2opt_to_tour(model, tour)

    # Calculate profit
    profit = model.compute_tour_profit(tour)

    return tour, profit


def farthest_insertion(
    model: VRPPModel, profit_aware_operators: bool = False, expand_pool: bool = False
) -> Tuple[List[int], float]:
    """
    Farthest insertion heuristic for VRPP (Section 4 of Fischetti et al.).

    Uses the farthest_profit_insertion operator to iteratively select the
    node farthest from the current tour that is still profitable to insert
    and add it to the tour.

    Args:
        model: VRPPModel instance.
        profit_aware_operators: Whether to use the profit-aware farthest insertion.
        expand_pool: Whether to consider all customer nodes for insertion.

    Returns:
        Tuple of (tour, profit).
    """
    # Start with mandatory nodes or empty
    if model.mandatory_nodes:
        initial_nodes = list(model.mandatory_nodes)
        routes = [[node] for node in initial_nodes]
    else:
        routes = [[]]

    # Get all customer nodes
    all_customers = list(range(1, model.n_nodes))

    # Filter out already included nodes
    remaining = [n for n in all_customers if n not in (model.mandatory_nodes or set())]

    if profit_aware_operators:
        routes = operator_farthest_profit_insertion(
            routes=routes,
            removed_nodes=remaining,
            dist_matrix=model.cost_matrix,
            wastes=model.wastes,
            capacity=model.capacity,
            R=model.R,
            C=model.C,
            mandatory_nodes=list(model.mandatory_nodes) if model.mandatory_nodes else None,
            expand_pool=expand_pool,
        )
    else:
        routes = operator_farthest_insertion(
            routes=routes,
            removed_nodes=remaining,
            dist_matrix=model.cost_matrix,
            wastes=model.wastes,
            capacity=model.capacity,
            mandatory_nodes=list(model.mandatory_nodes) if model.mandatory_nodes else None,
            expand_pool=expand_pool,
        )

    # Convert to single tour
    tour = _routes_to_tour(routes, model.depot)

    # Improve with 2-opt
    tour = _apply_2opt_to_tour(model, tour)

    profit = model.compute_tour_profit(tour)
    return tour, profit


def _routes_to_tour(routes: List[List[int]], depot: int) -> List[int]:
    """
    Convert multiple routes into a single tour with depot returns.

    Args:
        routes: List of routes (each route is a list of node indices).
        depot: Depot node index (typically 0).

    Returns:
        Single tour: [depot, route1_nodes..., depot, route2_nodes..., depot]
    """
    if not routes or all(len(r) == 0 for r in routes):
        return [depot, depot]

    tour = [depot]
    for route in routes:
        if route:  # Skip empty routes
            tour.extend(route)
            tour.append(depot)

    # Ensure tour ends at depot
    if tour[-1] != depot:
        tour.append(depot)

    return tour


def _apply_2opt_to_tour(model: VRPPModel, tour: List[int], max_iterations: int = 100) -> List[int]:
    """
    Apply 2-opt local search improvement to a tour.

    This is a lightweight Python implementation for the Branch-and-Cut heuristics.
    For more intensive optimization, use the k_opt operators from the operators module.

    Args:
        model: VRPPModel instance.
        tour: Initial tour.
        max_iterations: Maximum number of improvement iterations.

    Returns:
        Improved tour.
    """
    if len(tour) <= 3:
        return tour

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1])

                current_cost = model.cost_matrix[tour[i - 1], tour[i]] + model.cost_matrix[tour[j], tour[j + 1]]

                new_cost = model.cost_matrix[tour[i - 1], tour[j]] + model.cost_matrix[tour[i], tour[j + 1]]

                if new_cost < current_cost - 1e-6:
                    # Perform 2-opt swap: reverse tour[i:j+1]
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
                    break

            if improved:
                break

    return tour
