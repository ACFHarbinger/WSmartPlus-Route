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
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        # Calculate centroids of routes for proximity
        centroids = []
        for route in routes:
            # Exclude depot (0) for centroid calculation
            nodes = [n for n in route if n != 0]
            if nodes:
                center = coords.iloc[nodes][["Lat", "Lng"]].mean().values
                centroids.append(center)
            else:
                centroids.append(coords.iloc[0][["Lat", "Lng"]].values)

        # Try all possible seeds (routes)
        for i in range(len(routes)):
            # Find R-1 nearest neighbors to route i
            neighborhood_indices = find_route_neighbors(i, centroids, subproblem_size)

            # Form subproblem
            subproblem_nodes = []
            for idx in neighborhood_indices:
                subproblem_nodes.extend([n for n in routes[idx] if n != 0])

            if not subproblem_nodes:
                continue

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
                # In a real implementation, we'd replace the old routes with new ones
                # Here we just update the specific routes in the list
                for local_idx, global_idx in enumerate(neighborhood_indices):
                    if local_idx < len(new_routes):
                        routes[global_idx] = new_routes[local_idx]
                    else:
                        routes[global_idx] = [0, 0]  # Empty route
                improved = True
                break

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
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Hybrid Genetic Search (HGS)."""
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
        dist_matrix=distance_matrix,
        wastes=wastes_dict,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=must_go,
    )
    routes, profit, _ = solver.solve()
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
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Adaptive Large Neighborhood Search (ALNS)."""
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
        dist_matrix=distance_matrix,
        wastes=wastes_dict,
        capacity=capacity,
        R=R,
        C=C,
        params=params,
        mandatory_nodes=must_go,
    )
    routes, profit, _ = solver.solve()
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
