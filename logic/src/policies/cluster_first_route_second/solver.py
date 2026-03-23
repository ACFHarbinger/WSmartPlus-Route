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

from logic.src.policies.travelling_salesman_problem.tsp import calculate_tour_cost, find_route


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
    if num_clusters > 0:
        k = num_clusters
    else:
        # Dynamic calculation: as many clusters as necessary to fit total waste
        total_waste = sum(wastes.get(node, 0) for node in must_go)
        # At least 1 cluster if nodes exist, and at least n_vehicles
        min_k = math.ceil(total_waste / capacity) if capacity > 0 else 1
        k = max(n_vehicles, min_k)
        if k == 0 and must_go:
            k = 1

    # STAGE 1: CLUSTERING
    # Group bins by their angular coordinate relative to the depot, applying capacity/profit checks
    clusters = fisher_jaikumar_clustering(coords, must_go, k, wastes, capacity, R, C, distance_matrix)

    # STAGE 2: ROUTING
    # Solve a separate TSP for each cluster. Tours start and end at the depot (0).
    full_tour = [0]
    total_cluster_cost = 0.0
    for cluster in clusters:
        if not cluster:
            continue

        # Solve TSP for the current sector
        cluster_tour = find_route(distance_matrix, cluster, time_limit=time_limit, seed=seed)
        cluster_cost = calculate_tour_cost(distance_matrix, cluster_tour)
        total_cluster_cost += cluster_cost

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

    # 1. Depot and node feature extraction
    depot_lat, depot_lng = _get_depot_coords(coords)
    df_nodes = _compute_node_features(coords, must_go, depot_lat, depot_lng, distance_matrix)

    # 2. Seed Selection
    seeds = _select_initial_seeds(df_nodes, k)
    if not seeds:
        return []

    # 3. Assignment (GAP approximation)
    clusters = _perform_gap_assignment(seeds, must_go, wastes, capacity, distance_matrix)

    # Filter out empty clusters
    return [c for c in clusters if c]


def _get_depot_coords(coords: Any) -> Tuple[float, float]:
    """Extract coordinates of the depot node (index 0)."""
    if isinstance(coords, pd.DataFrame):
        return float(coords.iloc[0]["Lat"]), float(coords.iloc[0]["Lng"])
    return float(coords[0, 0]), float(coords[0, 1])


def _compute_node_features(
    coords: Any,
    must_go: List[int],
    depot_lat: float,
    depot_lng: float,
    distance_matrix: np.ndarray,
) -> pd.DataFrame:
    """Compute angular positions and distances for all nodes relative to depot."""
    node_data = []
    for idx in must_go:
        if isinstance(coords, pd.DataFrame):
            lat, lng = coords.iloc[idx]["Lat"], coords.iloc[idx]["Lng"]
        else:
            lat, lng = coords[idx, 0], coords[idx, 1]
        angle = math.atan2(lat - depot_lat, lng - depot_lng)
        dist = distance_matrix[0, idx]
        node_data.append({"idx": idx, "angle": angle, "dist": dist})

    return pd.DataFrame(node_data).sort_values("angle")


def _select_initial_seeds(df_nodes: pd.DataFrame, k: int) -> List[int]:
    """Partition space into K sectors and pick furthest node in each as seed."""
    seeds = []
    df_sorted = df_nodes.sort_values("angle")
    for i in range(k):
        start_angle = -math.pi + (i * 2 * math.pi / k)
        end_angle = -math.pi + ((i + 1) * 2 * math.pi / k)

        sector = df_sorted[(df_sorted["angle"] >= start_angle) & (df_sorted["angle"] < end_angle)]
        if not sector.empty:
            seed_idx = int(sector.loc[sector["dist"].idxmax(), "idx"])
            seeds.append(seed_idx)
        else:
            # Fallback if sector is empty: pick furthest unassigned node overall
            available = df_sorted[~df_sorted["idx"].isin(seeds)]
            if not available.empty:
                seeds.append(int(available.loc[available["dist"].idxmax(), "idx"]))

    return seeds


def _perform_gap_assignment(
    seeds: List[int],
    must_go: List[int],
    wastes: Dict[int, float],
    capacity: float,
    distance_matrix: np.ndarray,
) -> List[List[int]]:
    """Assign each node to the seed minimizing insertion cost, respecting capacity."""
    clusters: List[List[int]] = [[] for _ in range(len(seeds))]
    loads = [0.0] * len(seeds)

    # Sort nodes by distance to depot (furthest first)
    remaining_nodes = sorted(must_go, key=lambda x: distance_matrix[0, x], reverse=True)

    for node in remaining_nodes:
        if node in seeds:
            s_idx = seeds.index(node)
            clusters[s_idx].append(node)
            loads[s_idx] += wastes.get(node, 0)
            continue

        best_idx = _find_best_seed(node, seeds, loads, wastes, capacity, distance_matrix)

        if best_idx != -1:
            clusters[best_idx].append(node)
            loads[best_idx] += wastes.get(node, 0)
        else:
            # Force-assign to least-utilized cluster if it doesn't fit anywhere
            best_fallback = int(np.argmin(loads))
            clusters[best_fallback].append(node)
            loads[best_fallback] += wastes.get(node, 0)

    return clusters


def _find_best_seed(
    node: int,
    seeds: List[int],
    loads: List[float],
    wastes: Dict[int, float],
    capacity: float,
    distance_matrix: np.ndarray,
) -> int:
    """Find the cluster seed that minimizes insertion cost for the given node."""
    best_seed_idx = -1
    min_insertion = float("inf")
    waste = wastes.get(node, 0)

    for s_idx, seed in enumerate(seeds):
        # Fisher & Jaikumar cost: f(i, k) = d(0, i) + d(i, k) - d(0, k)
        insertion_cost = distance_matrix[0, node] + distance_matrix[node, seed] - distance_matrix[0, seed]

        if insertion_cost < min_insertion and loads[s_idx] + waste <= capacity:
            min_insertion = insertion_cost
            best_seed_idx = s_idx

    return best_seed_idx


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
