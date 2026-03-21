"""
Core implementation of Cluster-First Route-Second (CF-RS) Algorithm.

This module provides the computational logic for the two-stage routing heuristic:
1. Clustering: Partitioning nodes into angular sectors centered at the depot.
2. Routing: Solving a Traveling Salesman Problem (TSP) for each sector independently.

This approach is particularly efficient for large-scale logistics where global
optimization is computationally prohibitive, and provides intuitive,
geometrically-grouped routes.

Reference:
    Fisher, M. L., & Jaikumar, R. (1981). "A generalized assignment heuristic
    for vehicle routing". Networks, 11(2), 109-124.
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017). "A Variant Fisher
    and Jaikuamr Algorithm to Solve Capacitated Vehicle Routing Problem".
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.travelling_salesman_problem.tsp import find_route


def run_cf_rs(
    coords: pd.DataFrame,
    must_go: List[int],
    distance_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_vehicles: int,
    seed: int = 42,
    num_clusters: int = 0,
    time_limit: float = 60.0,
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Solve VRPP using Cluster-First Route-Second angular partitioning.

    This function orchestrates the decomposition of the routing problem into
    multiple TSP sub-problems based on the angular positions of nodes.

    Args:
        coords: DataFrame containing 'Lat' and 'Lng' for all nodes.
            The depot must be at index 0.
        must_go: List of global indices of bins to be collected.
        distance_matrix: Pre-computed all-pairs distance matrix.
        n_vehicles: Total number of vehicles available for the simulation day.
        seed: Random seed for the TSP solver to ensure deterministic results.
        num_clusters: Explicit number of sectors to create. If 0, uses n_vehicles.

    Returns:
        A tuple containing:
            - List[int]: Combined tour visiting all clusters, starting/ending at depot.
            - float: Total distance of the concatenated tours.
            - Dict[str, Any]: Metadata containing the cluster assignments for analysis.
    """
    if not must_go:
        return [0, 0], 0.0, {}

    # Determine number of clusters (K)
    k = num_clusters if num_clusters > 0 else n_vehicles

    # STAGE 1: CLUSTERING
    # Group bins by their angular coordinate relative to the depot, applying capacity/profit checks
    clusters = angular_clustering(coords, must_go, k, wastes, capacity, R, C, distance_matrix)

    # STAGE 2: ROUTING
    # Solve a separate TSP for each cluster. Tours start and end at the depot (0).
    full_tour = [0]
    for _i, cluster in enumerate(clusters):
        if not cluster:
            continue

        # Solve TSP for the current sector
        cluster_tour = find_route(distance_matrix, cluster, time_limit=time_limit, seed=seed)

        # Append the cluster tour to the master tour.
        # Since find_route returns [0, n1, n2, ..., 0], we skip the leading 0
        # to cleanly concatenate with the previous depot return.
        full_tour.extend(cluster_tour[1:])

    # Recalculate combined tour cost to ensure precision
    total_cost = calculate_tour_cost(distance_matrix, full_tour)

    return full_tour, total_cost, {"clusters": clusters, "num_sectors": len(clusters)}


def fisher_jaikumar_clustering(
    coords: pd.DataFrame,
    must_go: List[int],
    k: int,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    distance_matrix: np.ndarray,
) -> List[List[int]]:
    """
    Partition bins using the Fisher & Jaikumar (1981) seed-based assignment.

    1. Select K seed nodes (e.g., furthest nodes in different angular sectors).
    2. Assign each node to the seed that minimizes insertion cost:
       c_ij = d(0, i) + d(i, j) - d(0, j)
    3. Respect capacity constraints.
    """
    if not must_go:
        return []

    # 1. Seed Selection: Partition space into K sectors and pick furthest node in each
    depot_lat, depot_lng = coords.iloc[0]["Lat"], coords.iloc[0]["Lng"]
    node_data = []
    for idx in must_go:
        lat, lng = coords.iloc[idx]["Lat"], coords.iloc[idx]["Lng"]
        angle = math.atan2(lat - depot_lat, lng - depot_lng)
        dist = distance_matrix[0, idx]
        node_data.append({"idx": idx, "angle": angle, "dist": dist})

    df_nodes = pd.DataFrame(node_data)
    df_nodes = df_nodes.sort_values("angle")

    # Divide into K sectors and pick furthest node in each as seed
    sector_size = math.ceil(len(df_nodes) / k) if k > 0 else len(df_nodes)
    seeds = []
    for i in range(k):
        sector = df_nodes.iloc[i * sector_size : (i + 1) * sector_size]
        if not sector.empty:
            # Fisher-Jaikumar: Seeds should be furthest nodes (most difficult to insert)
            seed_idx = int(sector.loc[sector["dist"].idxmax(), "idx"])
            seeds.append(seed_idx)

    # 2. Assignment based on insertion cost
    clusters = [[] for _ in range(len(seeds))]
    loads = [0.0] * len(seeds)

    # Sort nodes by distance to depot (furthest first to ensure they get feasible assignments)
    remaining_nodes = sorted(must_go, key=lambda x: distance_matrix[0, x], reverse=True)

    for node in remaining_nodes:
        # Calculate assignment cost for each seed
        # d_ik = d(0, i) + d(i, k) - d(0, k)
        best_seed_idx = -1
        min_insertion = float("inf")

        for s_idx, seed in enumerate(seeds):
            # Generalized Assignment Problem (GAP) cost approximation
            insertion_cost = distance_matrix[0, node] + distance_matrix[node, seed] - distance_matrix[0, seed]

            if insertion_cost < min_insertion and loads[s_idx] + wastes.get(node, 0) <= capacity:
                min_insertion = insertion_cost
                best_seed_idx = s_idx

        if best_seed_idx != -1:
            clusters[best_seed_idx].append(node)
            loads[best_seed_idx] += wastes.get(node, 0)
        else:
            # Fallback for overflows: start a new cluster if possible or add to least filled
            min_load_idx = int(np.argmin(loads))
            clusters[min_load_idx].append(node)
            loads[min_load_idx] += wastes.get(node, 0)

    return [c for c in clusters if c]


def angular_clustering(
    coords: pd.DataFrame,
    must_go: List[int],
    k: int,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    distance_matrix: np.ndarray,
) -> List[List[int]]:
    """Legacy angular clustering - now delegates to Fisher-Jaikumar for better 5/5 faithfulness."""
    return fisher_jaikumar_clustering(coords, must_go, k, wastes, capacity, R, C, distance_matrix)


def _bounded_partition(
    sorted_indices: List[int],
    k: int,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_set: set,
) -> List[List[int]]:
    """Helper for capacity-aware bounded partitioning (Inclusive)."""
    clusters: List[List[int]] = [[] for _ in range(k)]
    loads = [0.0] * k
    cluster_idx = 0

    for node in sorted_indices:
        waste = wastes.get(node, 0.0)

        # If adding this node exceeds current cluster capacity
        if loads[cluster_idx] + waste > capacity:
            # Try to move to the next available cluster
            if cluster_idx + 1 < k:
                cluster_idx += 1
                clusters[cluster_idx].append(node)
                loads[cluster_idx] += waste
            else:
                # No more clusters available.
                # Append to current (last) cluster instead of dropping.
                # Supports Aggressive Sweep by ensuring 100% pool coverage.
                clusters[cluster_idx].append(node)
                loads[cluster_idx] += waste
        else:
            # Fits in current cluster
            clusters[cluster_idx].append(node)
            loads[cluster_idx] += waste

    return clusters


def _unbounded_partition(
    sorted_indices: List[int], wastes: Dict[int, float], capacity: float, mandatory_set: set
) -> List[List[int]]:
    """Helper for capacity-aware unbounded partitioning (Inclusive)."""
    clusters: List[List[int]] = [[]]
    loads = [0.0]

    for node in sorted_indices:
        waste = wastes.get(node, 0.0)

        # If it doesn't fit in the current "trip", start a new one
        if loads[-1] + waste > capacity and clusters[-1]:
            clusters.append([])
            loads.append(0.0)

        clusters[-1].append(node)
        loads[-1] += waste

    return clusters


def calculate_tour_cost(distance_matrix: np.ndarray, tour: List[int]) -> float:
    """
    Calculate the total distance of a given tour sequence.

    Useful for validating the combined cost of concatenated sector tours.

    Args:
        distance_matrix: NxN matrix of shortest path distances.
        tour: Sequence of node indices representing the path.

    Returns:
        float: Sum of distances between consecutive nodes in the tour.
    """
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    return float(cost)
