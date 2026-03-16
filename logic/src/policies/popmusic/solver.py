"""
Core implementation of POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions).

POPMUSIC is a matheuristic framework that decomposes a large combinatorial
optimization problem into subproblems (sets of routes) and iteratively
optimizes them.

Reference:
    Taillard, É., & Voss, S. (2002). "POPMUSIC: a metaheuristic for routing
    problems". Metaheuristics: theory, research and applications, 185-200.
"""

from random import Random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.other.operators.heuristics.nn_initialization import build_nn_routes
from logic.src.policies.travelling_salesman_problem.tsp import find_route, get_route_cost


def run_popmusic(
    coords: pd.DataFrame,
    must_go: List[int],
    distance_matrix: np.ndarray,
    n_vehicles: int,
    subproblem_size: int = 3,
    max_iterations: int = 100,
    base_solver: str = "fast_tsp",
    seed: int = 42,
    # New parameters for NN initialization
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = 1.0e9,
    R: float = 1.0,
    C: float = 0.0,
) -> Tuple[List[int], float, Dict[str, Any]]:
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
        seed: Random seed.
        wastes: Dictionary of node wastes.
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier for NN initialization.
        C: Cost multiplier for NN initialization.

    Returns:
        Combined tour, cost, and metadata.
    """
    if not must_go:
        return [0, 0], 0.0, {}

    # 1. INITIAL SOLUTION: Nearest Neighbor Initialization
    # build_nn_routes creates geographically compact routes based on capacity
    rng = Random(seed)
    wastes_dict = wastes if wastes is not None else {i: 0.0 for i in must_go}

    # We treat all must_go bins as mandatory for initialization
    initial_clusters = build_nn_routes(
        nodes=must_go,
        mandatory_nodes=must_go,
        wastes=wastes_dict,
        capacity=capacity,
        dist_matrix=distance_matrix,
        R=R,
        C=C,
        rng=rng,
    )

    # Limit number of routes to n_vehicles if necessary (though POPMUSIC handles n_vehicles)
    # If build_nn_routes produces more than n_vehicles, we might need a different handling,
    # but typically for VRPP/WCVRP we use what's generated.
    routes = []
    for cluster in initial_clusters:
        if cluster:
            # Add depot prefix/suffix and optimize the local cluster tour
            route = find_route(distance_matrix, cluster, seed=seed)
            routes.append(route)

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

            # Simple re-routing for now (could use ALNS if configured)
            new_tour = find_route(distance_matrix, subproblem_nodes, seed=seed)
            # Re-split (greedy simplified split)
            new_routes = split_tour(new_tour, len(neighborhood_indices), distance_matrix)
            new_cost = sum(get_route_cost(distance_matrix, r) for r in new_routes)

            if new_cost < old_cost - 1e-6:
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

    # Flatten routes into a single tour
    full_tour = [0]
    for r in routes:
        full_tour.extend(r[1:])

    total_cost = get_route_cost(distance_matrix, full_tour)

    return full_tour, total_cost, {"iterations": iteration, "num_routes": len(routes)}


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
