"""
Subgradient Optimization for LKH-3 π-penalties.

This module implements the Held-Karp subgradient ascent algorithm to find optimal
node penalties π that maximize the lower bound given by the length of the
minimum spanning 1-tree. These penalties are used to compute the α-measure
candidate sets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def compute_min_1_tree(distance_matrix: np.ndarray, pi: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute a Minimum 1-Tree for a given cost matrix and node penalties pi.

    A 1-tree is a tree with n edges consisting of a minimum spanning tree (MST)
    of nodes {1, ..., n-1} combined with two edges incident to node 0.

    Cost(i, j) = d(i, j) + pi(i) + pi(j).

    Args:
        distance_matrix: Symmetric cost matrix (n × n).
        pi: Node penalties array (size n).

    Returns:
        tuple: (1_tree_length, degrees, edges)
            - 1_tree_length: Length of the minimum 1-tree under penalized costs.
            - degrees: Degree of each node in the 1-tree.
            - edges: (n × 2) array of edges in the 1-tree.
    """
    n = len(distance_matrix)
    if n < 3:
        # Trivial case
        if n == 2:
            return float(distance_matrix[0, 1] + pi[0] + pi[1]), np.array([1, 1]), np.array([[0, 1]])
        return 0.0, np.zeros(n, dtype=int), np.empty((0, 2), dtype=int)

    # 1. Compute MST of nodes {1, ..., n-1}
    sub_dist = distance_matrix[1:, 1:]
    sub_pi = pi[1:]

    # Broadcast pi addition: D = d + pi_row + pi_col
    # Use float64 for intermediate calculations to prevent overflow/precision loss
    D = sub_dist + sub_pi[:, np.newaxis] + sub_pi[np.newaxis, :]

    mst_sparse = minimum_spanning_tree(D)
    mst_edges_coo = mst_sparse.tocoo()

    # 2. Find two cheapest edges from node 0 (depot) to nodes {1, ..., n-1}
    # D[0, j] = d[0, j] + pi[0] + pi[j]
    d0 = distance_matrix[0, 1:] + pi[0] + pi[1:]

    nearest_indices = np.argsort(d0)[:2]
    e1_idx = nearest_indices[0] + 1
    e2_idx = nearest_indices[1] + 1

    # 3. Aggregate results
    # Total length in penalized coordinates
    total_length = float(mst_sparse.sum() + d0[nearest_indices[0]] + d0[nearest_indices[1]])

    degrees = np.zeros(n, dtype=int)
    degrees[0] = 2
    degrees[e1_idx] += 1
    degrees[e2_idx] += 1

    edges_list = [(0, e1_idx), (0, e2_idx)]
    for u, v in zip(mst_edges_coo.row, mst_edges_coo.col):
        # Convert back to global indices (sub_dist was 1:)
        u_global, v_global = u + 1, v + 1
        edges_list.append((u_global, v_global))
        degrees[u_global] += 1
        degrees[v_global] += 1

    return total_length, degrees, np.array(edges_list)


def solve_subgradient(
    distance_matrix: np.ndarray,
    max_iterations: int = 200,
    n_original: Optional[int] = None,
    initial_pi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Held-Karp subgradient ascent to optimize node penalties pi.

    The algorithm attempts to find π that maximizes:
        L(π) = W(T_π) - 2 * sum(π_i)
    where W(T_π) is the minimum 1-tree length under penalized costs.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        max_iterations: Maximum ascent steps.
        n_original: Number of original nodes (excluding augmented dummy nodes).
        initial_pi: Optional starting penalties.

    Returns:
        Optimal node penalties pi (size n).
    """
    n = len(distance_matrix)
    if n < 3:
        return np.zeros(n)

    pi = initial_pi if initial_pi is not None else np.zeros(n)
    best_lb = -np.inf

    # Period constants (Helsgaun 2000 uses more complex decay)
    t = 1.0  # Step size factor

    for i in range(max_iterations):
        lb_pi, degrees, _ = compute_min_1_tree(distance_matrix, pi)

        # Real Lower Bound = W(T_pi) - 2 * sum(pi)
        current_lb = lb_pi - 2 * np.sum(pi)

        if current_lb > best_lb:
            best_lb = current_lb  # type: ignore[assignment]

        # Stopping criterion: all degrees are 2 (found Hamiltonian tour)
        G = degrees - 2
        norm_sq = np.sum(G**2)
        if norm_sq == 0:
            break

        # Step size adjustment
        # step = t * (1.1 * best_lb - current_lb) / norm_sq
        # Using 1.1 multiplier is a heuristic from Held & Karp
        step = t * (1.05 * max(best_lb, current_lb) - current_lb + 1e-4) / norm_sq
        pi += step * G

        # Lock the main depot penalty to 0
        pi[0] = 0.0
        # Lock all augmented dummy depots to 0
        if n_original is not None:
            pi[n_original:] = 0.0

        # Decay step size factor
        if i % 10 == 0:
            t *= 0.9

    return pi
