"""
Core implementation of POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions).

POPMUSIC is a matheuristic framework that decomposes a large combinatorial
optimization problem into subproblems (sets of routes) and iteratively
optimizes them.

Reference:
    Taillard, É., & Voss, S. (2002). "POPMUSIC: a metaheuristic for routing
    problems". Metaheuristics: theory, research and applications, 185-200.
"""

import inspect
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig
from logic.src.policies.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.hybrid_genetic_search.params import HGSParams
from logic.src.policies.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.other.operators.heuristics.greedy_initialization import build_greedy_routes
from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes
from logic.src.policies.travelling_salesman_problem.tsp import find_route, get_route_cost


def run_popmusic(  # noqa: C901
    coords: pd.DataFrame,
    must_go: List[int],
    distance_matrix: np.ndarray,
    n_vehicles: int,
    subproblem_size: int = 3,
    max_iterations: int = 100,
    base_solver: str = "fast_tsp",
    base_solver_config: Optional[Any] = None,
    cluster_solver: str = "fast_tsp",
    cluster_solver_config: Optional[Any] = None,
    initial_solver: str = "nearest_neighbor",
    seed: int = 42,
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = 1.0e9,
    R: float = 1.0,
    C: float = 0.0,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
) -> Tuple[List[List[int]], float, float, Dict[str, Any]]:
    """
    Solve VRPP using POPMUSIC matheuristic.

    Args:
        coords: Node coordinates.
        must_go: List of bin indices to collect.
        distance_matrix: Distances between nodes.
        n_vehicles: Number of vehicles available.
        subproblem_size: Number of neighboring routes to optimize (R).
        max_iterations: Max subproblem attempts.
        base_solver: Solver for subproblems.
        base_solver_config: Configuration for base_solver.
        cluster_solver: Solver for cluster initialization.
        cluster_solver_config: Configuration for cluster initialization solver.
        initial_solver: Initial solution generation method.
        seed: Random seed.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        vrpp: Whether to use expansion pool for VRPP.
        profit_aware_operators: Whether to use profit-aware operators.

    Returns:
        Tuple of (routes, total_cost, total_profit, info_dict).
    """
    if not must_go:
        return [[0, 0]], 0.0, 0.0, {}

    # 1. INITIAL SOLUTION
    wastes_dict = wastes if wastes is not None else {}

    if initial_solver == "greedy":
        initial_clusters = build_greedy_routes(
            dist_matrix=distance_matrix,
            wastes=wastes_dict,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=must_go,
            rng=Random(seed),
        )
    elif initial_solver == "nearest_neighbor":
        initial_clusters = build_nn_routes(
            nodes=list(range(1, len(distance_matrix))),
            mandatory_nodes=must_go,
            wastes=wastes_dict,
            capacity=capacity,
            dist_matrix=distance_matrix,
            R=R,
            C=C,
            rng=Random(seed),
        )
    else:
        raise ValueError(f"Unknown initial solver: {initial_solver}")

    # Convert clusters to initial routes using _optimize_subproblem
    routes = []
    for cluster in initial_clusters:
        if cluster:
            # Add depot prefix/suffix and optimize the local cluster tour
            # using _optimize_subproblem for consistency
            sub_routes, _ = _optimize_subproblem(
                base_solver=cluster_solver,
                base_solver_config=cluster_solver_config,
                subproblem_nodes=cluster,
                distance_matrix=distance_matrix,
                wastes_dict=wastes_dict,
                capacity=capacity,
                R=R,
                C=C,
                neighborhood_indices=[0],  # Dummy for initial
                must_go=must_go,
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
            # _optimize_subproblem returns List[List[int]], we expect one or more routes
            routes.extend(sub_routes)

    # 2. POPMUSIC ITERATIONS
    iteration = 0
    # Initialize global unassigned pool for VRPP support
    unassigned_nodes = set()
    # Initialize Active List (O-List) for POPMUSIC acceleration
    active_mask = [True] * len(routes)

    while any(active_mask) and iteration < max_iterations:
        # Calculate centroids of routes for proximity
        centroids = []
        for route in routes:
            # Exclude depot (0) for centroid calculation
            nodes = [n for n in route if n != 0]
            if nodes:
                # Use .loc to guarantee alignment with exact Node IDs
                center = coords.loc[nodes][["Lat", "Lng"]].mean().values
                centroids.append(center)
            else:
                # Use .loc for the depot as well
                centroids.append(coords.loc[0][["Lat", "Lng"]].values)

        # Try all possible seeds (routes)
        for i in range(len(routes)):
            # Skip inactive seeds (Active List optimization)
            if not active_mask[i]:
                continue

            # Skip ghost routes (routes without customer nodes)
            # This prevents wasting computation on depot-only routes before cleanup
            if not [n for n in routes[i] if n != 0]:
                continue

            # Find R-1 nearest neighbors to route i
            neighborhood_indices = find_route_neighbors(i, centroids, subproblem_size)

            # Form subproblem
            subproblem_nodes = []
            for idx in neighborhood_indices:
                subproblem_nodes.extend([n for n in routes[idx] if n != 0])

            if not subproblem_nodes:
                continue

            # Inject nearby unassigned nodes into the subproblem (VRPP support)
            if vrpp and unassigned_nodes and subproblem_nodes:
                nearby_unassigned = []

                # Calculate average edge length in the subproblem to use as a threshold
                sub_dists = [
                    distance_matrix[subproblem_nodes[i], subproblem_nodes[j]]
                    for i in range(len(subproblem_nodes))
                    for j in range(i + 1, len(subproblem_nodes))
                ]
                threshold = (np.mean(sub_dists) * 1.5) if sub_dists else distance_matrix.mean()

                for u_node in list(unassigned_nodes):
                    # Find minimum distance to any node in the active subproblem
                    min_dist_to_subproblem = min(distance_matrix[u_node, s_node] for s_node in subproblem_nodes)
                    if min_dist_to_subproblem <= threshold:
                        nearby_unassigned.append(u_node)

                # Remove from unassigned pool and add to subproblem
                for node in nearby_unassigned:
                    unassigned_nodes.remove(node)
                    subproblem_nodes.append(node)

            # Optimize subproblem
            old_cost = sum(get_route_cost(distance_matrix, routes[idx]) for idx in neighborhood_indices)
            old_rev = sum(wastes_dict.get(n, 0) * R for idx in neighborhood_indices for n in routes[idx] if n != 0)
            old_profit = old_rev - old_cost * C

            # Optimize subproblem using selected base_solver
            new_routes, new_profit = _optimize_subproblem(
                base_solver=base_solver,
                base_solver_config=base_solver_config,
                subproblem_nodes=subproblem_nodes,
                distance_matrix=distance_matrix,
                wastes_dict=wastes_dict,
                capacity=capacity,
                R=R,
                C=C,
                neighborhood_indices=neighborhood_indices,
                must_go=must_go,
                seed=seed,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
            )
            if new_profit > old_profit + 1e-6:
                # Update solution
                # Update existing routes in the neighborhood
                for local_idx, global_idx in enumerate(neighborhood_indices):
                    if local_idx < len(new_routes):
                        routes[global_idx] = new_routes[local_idx]
                        # Reactivate modified routes (Active List mechanism)
                        active_mask[global_idx] = True
                    else:
                        routes[global_idx] = [0, 0]  # Empty route
                        active_mask[global_idx] = True

                # CRITICAL: Append any extra routes created by the sub-solver
                # This prevents data loss when sub-solver creates more routes than the neighborhood size
                if len(new_routes) > len(neighborhood_indices):
                    extra_routes = new_routes[len(neighborhood_indices) :]
                    routes.extend(extra_routes)
                    # Extend active_mask for new routes (all marked as active)
                    active_mask.extend([True] * len(extra_routes))

                # Track unassigned nodes for VRPP support
                if vrpp:
                    # Find nodes from subproblem_nodes that are missing from new_routes
                    nodes_in_new_routes = set()
                    for route in new_routes:
                        nodes_in_new_routes.update([n for n in route if n != 0])

                    # Any nodes that were in the subproblem but not in new routes are unassigned
                    newly_unassigned = set(subproblem_nodes) - nodes_in_new_routes
                    unassigned_nodes.update(newly_unassigned)

                # Continue evaluating all seed routes in this iteration instead of breaking early
            else:
                # No improvement: deactivate the seed route (Active List mechanism)
                active_mask[i] = False

        # Clean up ghost routes while maintaining active_mask alignment
        valid_indices = []
        cleaned_routes = []

        for idx, r in enumerate(routes):
            # Keep route if it contains at least one non-depot customer node
            if [n for n in r if n != 0]:
                cleaned_routes.append(r)
                valid_indices.append(idx)

        # Sync routes and active_mask
        routes = cleaned_routes
        active_mask = [active_mask[idx] for idx in valid_indices]

        iteration += 1

    # Calculate final summary metrics across all routes
    total_cost = sum(get_route_cost(distance_matrix, r) for r in routes) * C
    total_rev = sum(wastes_dict.get(n, 0.0) * R for r in routes for n in r if n != 0)
    total_profit = total_rev - total_cost

    return routes, total_cost, total_profit, {"iterations": iteration, "num_routes": len(routes)}


def _optimize_subproblem(
    base_solver: Optional[str],
    base_solver_config: Optional[Any],
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    must_go: List[int],
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using the selected base_solver."""
    time_limit = 1.0

    # Extract the relevant config if it's nested (POPMUSICSubSolverConfig or dict)
    actual_config = base_solver_config
    if base_solver and base_solver_config:
        if hasattr(base_solver_config, base_solver):
            actual_config = getattr(base_solver_config, base_solver)
        elif isinstance(base_solver_config, dict) and base_solver in base_solver_config:
            actual_config = base_solver_config[base_solver]
            # Handle list-of-dicts style from Hydra configuration if necessary
            if isinstance(actual_config, list) and len(actual_config) > 0 and isinstance(actual_config[0], dict):
                actual_config = actual_config[0]

    if base_solver == "fast_tsp" or base_solver is None:
        return _optimize_with_fast_tsp(
            subproblem_nodes, distance_matrix, wastes_dict, capacity, R, C, actual_config, time_limit, seed, vrpp
        )
    elif base_solver == "hgs":
        return _optimize_with_hgs(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            neighborhood_indices,
            must_go,
            actual_config,
            time_limit,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    elif base_solver == "alns":
        return _optimize_with_alns(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            must_go,
            actual_config,
            time_limit,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    else:
        raise ValueError(f"Unsupported base_solver: {base_solver}")


def _optimize_with_fast_tsp(
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = False,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using FastTSP and greedy split."""
    actual_time_limit = time_limit
    if config and hasattr(config, "time_limit"):
        actual_time_limit = config.time_limit
    elif isinstance(config, dict):
        actual_time_limit = config.get("time_limit", actual_time_limit)

    new_tour = find_route(distance_matrix, subproblem_nodes, time_limit=actual_time_limit, seed=seed)
    # find_route returns [0, n1, n2, ..., 0]. LinearSplit expects the giant tour EXCLUDING depot.
    giant_tour = [n for n in new_tour if n != 0]

    # Re-split using LinearSplit (optimal capacity-aware splitting)
    splitter = LinearSplit(
        dist_matrix=distance_matrix,
        wastes=wastes_dict,
        capacity=capacity,
        R=R,
        C=C,
        vrpp=vrpp,
    )
    new_routes, new_profit = splitter.split(giant_tour)
    return new_routes, new_profit


def _optimize_with_hgs(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    must_go: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Hybrid Genetic Search (HGS)."""
    # Ensure subproblem isolation: only optimize nodes in the subproblem
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for HGS optimization")

    # Create local must_go: intersection of global must_go and subproblem_nodes
    local_must_go = [n for n in must_go if n in subproblem_nodes]

    # Build local distance matrix: depot (0) + subproblem_nodes
    local_nodes = [0] + subproblem_nodes
    node_to_local_idx = {node: i for i, node in enumerate(local_nodes)}

    local_dist_matrix = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local_idx[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_must_go_indices = [node_to_local_idx[n] for n in local_must_go]

    if isinstance(config, HGSConfig):
        params = HGSParams.from_config(config)
    elif isinstance(config, dict):
        # Filter for only valid HGSParams fields to avoid errors
        valid_keys = set(inspect.signature(HGSParams).parameters.keys())
        params_dict = {k: v for k, v in config.items() if k in valid_keys}
        params = HGSParams(**params_dict)
    else:
        params = HGSParams(
            max_vehicles=len(neighborhood_indices),
            time_limit=time_limit,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

    # Ensure max_vehicles and time_limit are set correctly if not in config
    if params.max_vehicles <= 0:
        params.max_vehicles = len(neighborhood_indices)
    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = HGSSolver(
        dist_matrix=local_dist_matrix,
        wastes=local_wastes,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=local_must_go_indices,
    )
    local_routes, profit, _ = solver.solve()

    # Map local routes back to global node IDs
    routes = []
    for local_route in local_routes:
        global_route = [local_nodes[i] if i < len(local_nodes) else 0 for i in local_route]
        routes.append(global_route)

    return routes, profit


def _optimize_with_alns(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    must_go: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Adaptive Large Neighborhood Search (ALNS)."""
    # Ensure subproblem isolation: only optimize nodes in the subproblem
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for ALNS optimization")

    # Create local must_go: intersection of global must_go and subproblem_nodes
    local_must_go = [n for n in must_go if n in subproblem_nodes]

    # Build local distance matrix: depot (0) + subproblem_nodes
    local_nodes = [0] + subproblem_nodes
    node_to_local_idx = {node: i for i, node in enumerate(local_nodes)}

    local_dist_matrix = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local_idx[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_must_go_indices = [node_to_local_idx[n] for n in local_must_go]

    if config:
        if isinstance(config, ALNSConfig):
            params = ALNSParams.from_config(config)
        elif isinstance(config, dict):
            # Filter for only valid ALNSParams fields
            valid_keys = set(inspect.signature(ALNSParams).parameters.keys())
            params_dict = {k: v for k, v in config.items() if k in valid_keys}
            params = ALNSParams(**params_dict)
        else:
            params = ALNSParams(
                max_iterations=1000,
                time_limit=time_limit,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
                seed=seed,
            )
    else:
        params = ALNSParams(
            max_iterations=1000,
            time_limit=time_limit,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

    # Ensure time_limit is set correctly if not in config
    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = ALNSSolver(
        dist_matrix=local_dist_matrix,
        wastes=local_wastes,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=local_must_go_indices,
    )
    local_routes, profit, _ = solver.solve()

    # Map local routes back to global node IDs
    routes = []
    for local_route in local_routes:
        global_route = [local_nodes[i] if i < len(local_nodes) else 0 for i in local_route]
        routes.append(global_route)

    return routes, profit


def find_route_neighbors(seed_idx: int, centroids: List[np.ndarray], k: int) -> List[int]:
    """Find k nearest route indices to the seed route based on centroids."""
    if len(centroids) <= k:
        return list(range(len(centroids)))

    seed_pos = centroids[seed_idx]
    distances = []
    for i, pos in enumerate(centroids):
        dist = np.linalg.norm(seed_pos - pos)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def split_tour(tour: List[int], k: int, distance_matrix: np.ndarray) -> List[List[int]]:
    """Simple greedy split of a giant tour into k routes."""
    nodes = [n for n in tour if n != 0]
    if not nodes:
        return [[0, 0] for _ in range(k)]

    n = len(nodes)
    size = n // k
    remainder = n % k

    routes = []
    start = 0
    for i in range(k):
        batch_size = size + (1 if i < remainder else 0)
        route_nodes = nodes[start : start + batch_size]
        if route_nodes:
            # For POPMUSIC simplicity, we just use the order from the giant tour
            routes.append([0] + route_nodes + [0])
        else:
            routes.append([0, 0])
        start += batch_size
    return routes
