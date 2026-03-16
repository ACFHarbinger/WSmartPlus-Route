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
    n_vehicles: int,
    seed: int = 42,
    num_clusters: int = 0,
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
    # Group bins by their angular coordinate relative to the depot
    clusters = angular_clustering(coords, must_go, k)

    # STAGE 2: ROUTING
    # Solve a separate TSP for each cluster. Tours start and end at the depot (0).
    full_tour = [0]

    for _i, cluster in enumerate(clusters):
        if not cluster:
            continue

        # Solve TSP for the current sector
        cluster_tour = find_route(distance_matrix, cluster, seed=seed)

        # Append the cluster tour to the master tour.
        # Since find_route returns [0, n1, n2, ..., 0], we skip the leading 0
        # to cleanly concatenate with the previous depot return.
        full_tour.extend(cluster_tour[1:])

    # Recalculate combined tour cost to ensure precision
    total_cost = calculate_tour_cost(distance_matrix, full_tour)

    return full_tour, total_cost, {"clusters": clusters, "num_sectors": len(clusters)}


def angular_clustering(coords: pd.DataFrame, must_go: List[int], k: int) -> List[List[int]]:
    """
    Partition bins into k angular sectors based on their position relative to the depot.

    This implements the 'Cluster-First' stage using polar angle sorting.
    It splits the 360-degree space around the depot into sectors such that
    each sector contains a balanced number of nodes.

    Args:
        coords: DataFrame with 'Lat' and 'Lng'. Depot is at index 0.
        must_go: Global node indices of the collection targets.
        k: Maximum number of clusters (sectors) to create.

    Returns:
        List[List[int]]: A list of K clusters, where each cluster is a list of node indices.
    """
    if k <= 1 or len(must_go) < k:
        return [must_go]

    # Reference point is the depot
    depot_lat = coords.iloc[0]["Lat"]
    depot_lng = coords.iloc[0]["Lng"]

    # Calculate polar angles (in radians) for all bins relative to the depot
    bin_angles = []
    for idx in must_go:
        lat = coords.iloc[idx]["Lat"]
        lng = coords.iloc[idx]["Lng"]
        # math.atan2 provides angle in range (-pi, pi]
        angle = math.atan2(lat - depot_lat, lng - depot_lng)
        bin_angles.append((idx, angle))

    # Sort bins by their angular position to ensure contiguous sector partitioning
    bin_angles.sort(key=lambda x: x[1])
    sorted_indices = [x[0] for x in bin_angles]

    # Partition the sorted list into k roughly equal-sized clusters
    clusters = []
    n = len(sorted_indices)
    base_size = n // k
    remainder = n % k

    start = 0
    for i in range(k):
        # Distribute the remainder nodes across the first clusters to maintain balance
        cluster_size = base_size + (1 if i < remainder else 0)
        clusters.append(sorted_indices[start : start + cluster_size])
        start += cluster_size

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
