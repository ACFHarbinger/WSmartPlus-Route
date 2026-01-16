"""
Single-vehicle routing utilities module.

This module provides helper functions for solving single-vehicle routing problems
including TSP construction, local search optimization, and capacity management.

Key Functions:
--------------
- find_route: Solve TSP using fast_tsp library
- local_search_2opt: Improve tours with 2-opt local search
- get_multi_tour: Split single tour into multiple depot trips (capacity aware)
- get_route_cost: Calculate total tour distance
- get_partial_tour: Remove bins to meet capacity constraints
- dist_matrix_from_graph: Compute all-pairs shortest paths from NetworkX graph

These utilities are used by higher-level policies (Regular, LastMinute, etc.)
to generate and optimize single-vehicle collection routes.
"""

from typing import List, Tuple

import fast_tsp
import networkx as nx
import numpy as np
import torch


def find_route(C, to_collect):
    """
    Solve TSP for a subset of nodes using the fast_tsp library.

    Constructs a tour visiting all nodes in to_collect, starting and ending at depot (0).
    Uses fast_tsp's heuristic TSP solver for quick solutions.

    Args:
        C (np.ndarray): Distance matrix (N x N) with depot at index 0
        to_collect (array-like): Node IDs to visit (excluding depot)

    Returns:
        List[int]: Tour starting and ending at depot. Format: [0, node1, node2, ..., 0]
    """
    to_collect_tmp = [0] + list(to_collect)
    tmpC = C[to_collect_tmp, :][:, to_collect_tmp]
    tour = fast_tsp.find_tour(tmpC)
    zero_index = tour.index(0)
    tour = tour[zero_index:] + tour[:zero_index]
    # cost = fast_tsp.compute_cost(tour, tmpC)
    tour2 = []
    for ii in range(0, len(tour) - 1):
        current_node = to_collect_tmp[tour[ii]]
        next_node = to_collect_tmp[tour[ii + 1]]
        tour2.append(current_node)
    tour2.extend([next_node, 0])
    return tour2


def local_search_2opt(tour, distance_matrix, max_iterations=200):
    """
    Improve tour using 2-opt local search (vectorized with NumPy).

    Iteratively swaps edge pairs to reduce total tour distance. Uses vectorized
    NumPy operations for performance. Stops when no improving move is found.

    Args:
        tour (array-like or torch.Tensor): Tour to improve (must start/end at depot)
        distance_matrix (np.ndarray or torch.Tensor): Distance matrix
        max_iterations (int): Maximum number of improvement rounds. Default: 200

    Returns:
        List[int]: Improved tour (same format as input)
    """
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()
    if torch.is_tensor(distance_matrix):
        distance_matrix = distance_matrix.cpu().numpy()

    best_tour = np.array(tour)
    n = len(best_tour)
    if n < 4:
        return best_tour.tolist()

    # Ensure it starts and ends at depot (0)
    if best_tour[0] != 0 or best_tour[-1] != 0:
        return best_tour.tolist()

    for _ in range(max_iterations):
        # i indices from 1 to n-3, j indices from i+1 to n-2
        i = np.arange(1, n - 2)
        j = np.arange(2, n - 1)

        idx_i, idx_j = np.meshgrid(i, j, indexing="ij")
        mask = idx_j > idx_i

        if not np.any(mask):
            break

        I_vals = idx_i[mask]
        J_vals = idx_j[mask]

        # Tour nodes at relevant indices
        t_prev_i = best_tour[I_vals - 1]
        t_curr_i = best_tour[I_vals]
        t_curr_j = best_tour[J_vals]
        t_next_j = best_tour[J_vals + 1]

        # Gain calculation: current_dist - new_dist
        d_curr = distance_matrix[t_prev_i, t_curr_i] + distance_matrix[t_curr_j, t_next_j]
        d_next = distance_matrix[t_prev_i, t_curr_j] + distance_matrix[t_curr_i, t_next_j]
        gains = d_curr - d_next

        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        if best_gain > 1e-5:
            # Apply the best edge swap found in this iteration
            target_i = I_vals[best_idx]
            target_j = J_vals[best_idx]
            best_tour[target_i : target_j + 1] = best_tour[target_i : target_j + 1][::-1]
        else:
            break

    return best_tour.tolist()


def get_route_cost(distancesC, tour):
    """
    Calculate total distance cost of a tour.

    Sums the edge distances along the tour path.
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        distancesC (np.ndarray or torch.Tensor): Distance matrix
        tour (list or np.ndarray or torch.Tensor): Sequence of node IDs

    Returns:
        float: Total tour distance
    """
    if isinstance(tour, torch.Tensor) and isinstance(distancesC, torch.Tensor):
        return distancesC[tour[:-1], tour[1:]].sum().cpu().numpy().item()
    else:
        distancesC2 = distancesC.copy() if isinstance(distancesC, np.ndarray) else np.array(distancesC)
        tour2 = tour.copy() if isinstance(tour, np.ndarray) else np.array(tour)
        return np.sum(distancesC2[tour2[:-1], tour2[1:]]).item()


def get_path_cost(G, p):
    """
    Calculate path cost in a NetworkX graph.

    Args:
        G (networkx.Graph): Graph with edge weights
        p (List[int]): Path as sequence of node IDs

    Returns:
        float: Total path cost (sum of edge weights)
    """
    last_node = p[0]
    c = 0
    for id_i in range(1, len(p)):
        try:
            c += G.get_edge_data(last_node, p[id_i])["weight"]
        except Exception:
            c += 1
        last_node = p[id_i]
    return c


def get_multi_tour(tour, bins_waste, max_capacity, distance_matrix):
    """
    Insert depot return trips to satisfy vehicle capacity constraints.

    Given a TSP tour that may violate capacity, inserts depot visits (0) whenever
    cumulative load would exceed max_capacity. This converts a single long tour
    into multiple depot round-trips.

    Args:
        tour (List[int]): Initial TSP tour (may violate capacity)
        bins_waste (np.ndarray): Waste amounts for each bin
        max_capacity (float): Vehicle capacity limit
        distance_matrix (np.ndarray): Distance matrix (for future cost calculation)

    Returns:
        List[int]: Modified tour with depot returns inserted. Format: [0, ..., 0, ..., 0]
    """
    depot_trips = 0
    final_tour = tour
    vehicle_collected = 0
    tmp_tour = [x - 1 for x in tour if x != 0]
    for i in range(len(tmp_tour)):
        cur_bin = tmp_tour[i]
        col_waste = bins_waste[cur_bin]
        if vehicle_collected + col_waste < max_capacity:
            vehicle_collected += col_waste
        elif vehicle_collected + col_waste > max_capacity:
            final_tour.insert(i + depot_trips, 0)
            vehicle_collected = col_waste
            depot_trips += 1
            # cost += distance_matrix[tmp_tour[i - 1], 0] + distance_matrix[0, cur_bin]
        else:
            final_tour.insert(i + depot_trips - 1, 0)
            vehicle_collected = 0
            depot_trips += 1
            # if i < len(tmp_tour) - 1:
            # cost += distance_matrix[cur_bin, 0] + distance_matrix[0, tmp_tour[i + 1]]
    return final_tour


def get_partial_tour(
    tour: List[int],
    bins: np.ndarray,
    max_capacity: float,
    distance_matrix: np.ndarray,
    cost: float,
) -> Tuple[np.ndarray, float]:
    """
    Reduce a tour to fit within vehicle capacity by removing bins with minimal waste.

    Args:
        tour (List[int]): Current tour.
        bins (np.ndarray): Waste amounts for each bin.
        max_capacity (float): Vehicle capacity limit.
        distance_matrix (np.ndarray): Distance matrix.
        cost (float): Current routing cost.

    Returns:
        Tuple[np.ndarray, float]: (Reduced tour, updated cost).
    """
    tmp_tour = np.array([x - 1 for x in tour if x != 0])
    total_waste = np.sum(bins[tmp_tour])
    while total_waste > max_capacity:
        min_waste_bin_idx = np.argmin(bins[tmp_tour])
        bin_to_remove = tmp_tour[min_waste_bin_idx]
        total_waste -= bins[bin_to_remove]
        cost -= float(distance_matrix[tmp_tour[min_waste_bin_idx - 1], bin_to_remove])
        tmp_tour = np.delete(tmp_tour, min_waste_bin_idx)
    return tmp_tour, cost


# Create matrix will all distances
def dist_matrix_from_graph(G: nx.Graph) -> Tuple[np.ndarray, List[List[List[int]]]]:
    """
    Compute all-pairs shortest path distances and paths from a NetworkX graph.

    Args:
        G (nx.Graph): Input graph with nodes 0..N-1 and weighted edges.

    Returns:
        Tuple[np.ndarray, List[List[List[int]]]]: (Distance matrix, Path matrix).
            Distance matrix is N x N numpy array of shortest path lengths.
            Path matrix contains the sequence of nodes for each shortest path.
    """
    paths_between_states = []
    n_vertices = len(G.nodes)
    dist_matrix = np.zeros((n_vertices, n_vertices), int)
    for id_i in range(n_vertices):
        paths_between_states.append([])
        for id_j in range(n_vertices):
            if id_i == id_j:
                paths_between_states[id_i].append([])
                continue
            p = nx.dijkstra_path(G, source=id_i, target=id_j)
            paths_between_states[id_i].append(p)
            dist_matrix[id_i, id_j] = int(get_path_cost(G, p))
    return dist_matrix, paths_between_states
