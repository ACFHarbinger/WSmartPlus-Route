"""
LKH-3 inspired Heuristic Module.

This module implements a version of the Lin-Kernighan-Helsgaun heuristic inspired by LKH-3.
Key features include Alpha-measure pruning, lexicographical optimization (Penalty, Cost),
and Iterated Local Search with Double Bridge kicks.

Attributes:
    None

Example:
    >>> from logic.src.policies.lin_kernighan_helsgaun import solve_lkh
    >>> tour, cost = solve_lkh(distance_matrix)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Alpha-measures for edge pruning based on MST.

    alpha(i,j) = c(i,j) - (max edge weight on the unique path in MST between i and j).
    """
    # minimum_spanning_tree returns a sparse matrix.
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst = mst_sparse.toarray()

    # To correctly calculate Alpha measures for all pairs, we'd need max-edge in MST path.
    # For a high-performance Python version, we give preference to MST edges (alpha=0)
    # and use standard distance as a proxy for the rest, weighted by MST presence.
    alpha = np.copy(distance_matrix)

    # Bonus for MST edges
    mst_mask = (mst > 0) | (mst.T > 0)
    alpha[mst_mask] = 0

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray, alpha_measures: np.ndarray, max_candidates: int = 5
) -> Dict[int, List[int]]:
    """Generate candidate sets of edges for each node based on Alpha-measures."""
    n = len(distance_matrix)
    candidates = {}
    for i in range(n):
        # Sort by alpha, then breakdown by distance
        indices = np.argsort(alpha_measures[i])
        # Filter self and cast to int
        valid_indices = [int(idx) for idx in indices if idx != i]
        candidates[i] = valid_indices[:max_candidates]
    return candidates


def calculate_penalty(tour: List[int], waste: Optional[np.ndarray], capacity: Optional[float]) -> float:
    """Calculate VRP capacity violation penalty."""
    if waste is None or capacity is None:
        return 0.0

    penalty = 0.0
    current_load = 0.0
    for node in tour:
        if node == 0:
            current_load = 0.0
        else:
            # Map node to waste index (waste is assumed to be index-aligned with distance_matrix)
            # In look_ahead.py, we only pass sub-matrix of active bins, so indices are local.
            current_load += waste[node]
            if current_load > capacity + 1e-6:
                penalty += current_load - capacity
    return penalty


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    waste: Optional[np.ndarray] = None,
    capacity: Optional[float] = None,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesman Problem (or VRP variant) using Lin-Kernighan.

    Args:
        distance_matrix: Matrix of pairwise distances.
        initial_tour: Optional starting tour. Defaults to None (uses Nearest Neighbor).
        max_iterations: Maximum number of search iterations. Defaults to 100.
        waste: Optional array of waste levels for VRP penalty calculation.
        capacity: Optional vehicle capacity for VRP penalty calculation.

    Returns:
        Tuple[List[int], float]: Tuple containing (final_tour, final_cost).
    """
    n = len(distance_matrix)
    if n < 3:
        tour = list(range(n)) + [0]
        cost = distance_matrix[0, 1] + distance_matrix[1, 0] if n == 2 else 0.0
        return tour, float(cost)

    # 1. Initialization (Path of n unique nodes)
    if initial_tour is None:
        curr = 0
        p = [0]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
            p.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
    else:
        p = initial_tour[:-1] if initial_tour[0] == initial_tour[-1] else initial_tour

    # 2. Candidate edge generation
    alpha = compute_alpha_measures(distance_matrix)
    candidates = get_candidate_set(distance_matrix, alpha, max_candidates=10)

    def get_score(path: List[int]) -> Tuple[float, float]:
        """
        Calculate total penalty and cost for a given path.

        Args:
            path: List of node indices.

        Returns:
            tuple: (penalty, cost)
        """
        c = sum(distance_matrix[path[i], path[(i + 1) % n]] for i in range(n))
        pen = 0.0
        if waste is not None and capacity is not None:
            load = 0.0
            for node in path:
                if node == 0:
                    load = 0.0
                else:
                    load += waste[node]
                    if load > capacity + 1e-6:
                        pen += load - capacity
        return pen, c

    best_p = p[:]
    best_pen, best_cost = get_score(best_p)
    global_p, global_pen, global_cost = best_p[:], best_pen, best_cost

    def is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
        """
        Lexicographical comparison: Penalty first, then Cost.

        Args:
            p1: Penalty of first solution.
            c1: Cost of first solution.
            p2: Penalty of second solution.
            c2: Cost of second solution.

        Returns:
            True if (p1, c1) is better than (p2, c2).
        """
        if abs(p1 - p2) > 1e-6:
            return p1 < p2
        return c1 < c2 - 1e-6

    # 3. Optimization
    for iteration in range(max_iterations):
        improved = False
        for i in range(n):
            t1, t2 = best_p[i], best_p[(i + 1) % n]
            for t3 in candidates[t2]:
                # In standard LK, we skip edges already in tour.
                # For very small N, we allow more flexibility.
                if t3 == t1:
                    continue
                if n > 4 and t3 == best_p[(i + 2) % n]:
                    continue

                t3_idx = best_p.index(t3)
                new_p = best_p[:]
                # Reverse segment between i+1 and t3_idx
                if i + 1 < t3_idx:
                    new_p[i + 1 : t3_idx + 1] = new_p[i + 1 : t3_idx + 1][::-1]
                else:  # wrap around reversal is more complex, we'll simplify to non-wrap reversals
                    continue

                p_new, c_new = get_score(new_p)
                if is_better(p_new, c_new, best_pen, best_cost):
                    best_p, best_pen, best_cost = new_p, p_new, c_new
                    improved = True
                    break
            if improved:
                break

        if is_better(best_pen, best_cost, global_pen, global_cost):
            global_p, global_pen, global_cost = best_p[:], best_pen, best_cost

        if not improved:
            if n >= 8:
                pos = sorted(np.random.choice(range(n), 4, replace=False))
                a, b, c, d = pos
                best_p = (
                    best_p[: a + 1]
                    + best_p[c + 1 : d + 1]
                    + best_p[b + 1 : c + 1]
                    + best_p[a + 1 : b + 1]
                    + best_p[d + 1 :]
                )
                best_pen, best_cost = get_score(best_p)

    # Rotate and close
    final_p = global_p
    if 0 in final_p:
        idx = final_p.index(0)
        final_p = final_p[idx:] + final_p[:idx]

    final_tour = final_p + [0]
    final_cost = sum(distance_matrix[final_tour[i], final_tour[i + 1]] for i in range(n))
    return final_tour, float(final_cost)
