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
    """
    Partition bins into angular sectors based on their position relative to the depot,
    while enforcing capacity constraints and ensuring profitability of optional nodes.
    Creates as many clusters as needed to maintain clean sector independence.

    Args:
        coords: DataFrame with 'Lat' and 'Lng'. Depot is at index 0.
        must_go: Global node indices of the collection targets.
        k: Maximum number of clusters (sectors) to create. (Present for legacy API compatibility, but unbounded).
        wastes: Demand of each node.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        distance_matrix: Distance matrix.

    Returns:
        List[List[int]]: A list of clusters.
    """
    mandatory_set = set(must_go)
    pool = []

    # 1. Gather Candidate Pool
    n_nodes = len(distance_matrix) - 1
    for node in range(1, n_nodes + 1):
        if node in mandatory_set:
            pool.append(node)
        else:
            waste = wastes.get(node, 0.0)
            revenue = waste * R
            one_way_cost = distance_matrix[0, node] * C
            # Node-level Hurdle: Aggressive pool gathering for sweeping.
            # Revenue only needs to cover 5% of one-way cost to be considered.
            if revenue > 0.05 * one_way_cost:
                pool.append(node)

    if not pool:
        return []

    # Reference point is the depot
    depot_lat = coords.iloc[0]["Lat"]
    depot_lng = coords.iloc[0]["Lng"]

    # Calculate polar angles (in radians) for all pool bins relative to the depot
    bin_angles = []
    for idx in pool:
        lat = coords.iloc[idx]["Lat"]
        lng = coords.iloc[idx]["Lng"]
        angle = math.atan2(lat - depot_lat, lng - depot_lng)
        bin_angles.append((idx, angle))

    # Sort bins by their angular position for contiguous sector sweeping
    bin_angles.sort(key=lambda x: x[1])
    sorted_indices = [x[0] for x in bin_angles]

    # 3. Partitioning
    if k > 0:
        clusters = _bounded_partition(sorted_indices, k, wastes, capacity, mandatory_set)
    else:
        clusters = _unbounded_partition(sorted_indices, wastes, capacity, mandatory_set)

    # 4. Cluster Pruning: ensure clusters are meaningful
    final_clusters = []
    for cluster in clusters:
        if not cluster:
            continue

        # Mandatory nodes must always be collected
        if any(node in mandatory_set for node in cluster):
            final_clusters.append(cluster)
            continue

        cluster_waste = sum(wastes.get(node, 0.0) for node in cluster)
        cluster_revenue = cluster_waste * R

        # Estimate trip cost: depot to furthest node and back
        max_dist = max(distance_matrix[0, node] for node in cluster)
        est_cost = 2.0 * max_dist * C

        # Opportunistic Cluster Hurdle:
        # Revenue must cover 1.05x of round-trip OR cluster is 40% full.
        # This encourages more thorough routes to prevent future trips.
        if (cluster_revenue > 1.05 * est_cost) or (cluster_waste > 0.4 * capacity):
            final_clusters.append(cluster)

    return final_clusters


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
