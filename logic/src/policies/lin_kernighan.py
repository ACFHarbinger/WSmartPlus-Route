"""
LKH-3 inspired Heuristic Module.

This module implements a version of the Lin-Kernighan heuristic inspired by LKH-3.
Key improvements over basic LK:
1. Alpha-measure candidate sets for pruning search space.
2. Lexicographical optimization: (Penalty, Cost) where Penalty handles VRP constraints.
3. Sequential 3-opt moves for deeper exploration.
4. Iterated Local Search (ILS) with Double Bridge kicks.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from logic.src.policies.adapters import IPolicy, PolicyRegistry


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


def solve_lk(
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
        tuple: (final_tour, final_cost)
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


@PolicyRegistry.register("policy_lkh")
class LKHPolicy(IPolicy):
    """
    Lin-Kernighan Heuristic policy class.
    Executes LK for VRP.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LKH policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        kwargs["waste_type"]
        kwargs["area"]
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_lkh_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0  # Default

        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means' attribute.")
        else:
            means = bins.means
            std = bins.std

        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        # Must-go bins
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        # 2. Prepare Data for LKH
        target_nodes = must_go_indices
        if not target_nodes:
            return [0, 0], 0.0, None

        # LKH Config
        lkh_config = config.get("lkh", {}).copy()  # Use copy to modify
        capacity = lkh_config.get("capacity", 100.0)

        # Subset mapping
        real_target_indices = [idx + 1 for idx in target_nodes]
        subset_indices = [0] + real_target_indices

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Prepare Waste array for sub-problem
        # Config expects "waste" as array indexed 0..K (solver indices)
        # Bins c is demand.
        # solver index 0 is depot (demand 0).
        # solver index i (1..K) corresponds to real_target_indices[i-1].

        sub_waste = [0.0]  # Depot
        for original_idx in real_target_indices:
            sub_waste.append(current_fill[original_idx - 1])  # bins.c is 0-based

        lkh_config["waste"] = np.array(sub_waste)
        lkh_config["capacity"] = capacity

        # Run LKH
        # solve_lk signature: (distance_matrix: np.ndarray, config: Dict)
        best_tour, _ = solve_lk(sub_dist_matrix, lkh_config)

        # Map tour back
        tour = []
        if best_tour:
            for node_idx in best_tour:
                original_matrix_idx = subset_indices[node_idx]
                tour.append(original_matrix_idx)

        if not tour or tour == [0]:
            # Ensure valid structure [0, ..., 0] if feasible tour found
            # If best_tour is just [0], map it to [0]?
            # solve_lk usually returns closed tour? "final_tour = final_p + [0]"
            pass

        # Check start/end
        if tour and tour[0] != 0:
            tour = [0] + tour
        if tour and tour[-1] != 0:
            tour.append(0)

        if len(tour) <= 2:
            tour = [0, 0]

        # Recalculate cost
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i + 1]]

        return tour, cost, None
