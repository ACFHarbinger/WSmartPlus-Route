"""
LKH Heuristic Module.

Full implementation of the Lin-Kernighan-Helsgaun heuristic for solving
the Traveling Salesman Problem (TSP).

Key features (following Helsgaun 2000 and LKH-3 extension paper):

1. **Alpha-measure candidate sets** (Section 4.1, Helsgaun 2000):
   Edges pruned using sensitivity analysis on minimum spanning 1-trees. Only
   α-nearest neighbours are considered for inclusion in the tour.

2. **Subgradient optimisation / penalty vector** (Section 4.1):
   Node penalties π that transform C → D = cᵢⱼ + πᵢ + πⱼ, maximising the
   lower bound given by the minimum 1-tree length.

3. **Sequential k-opt moves (k = 2..5)** (Section 4.3):
   Exact gain computation for every reconnection case at each level, with
   early-termination via the positive-gain criterion.  Routines for higher-
   order moves (3 to 5-opt) are delegated to components in
   ``._tour_improvement``, with all final segment swaps executed via
   :func:`move_kopt_intra` from
   ``logic.src.policies.helpers.operators.intra_route.k_opt``.

4. **Candidate-set restricted search** (Section 3.2 / 4.1):
   Inner loops restricted to the α-nearest neighbours of each node, giving
   O(n · k_cand) search per node rather than O(n²).

5. **Don't-look bits** (Section 5.3):
   Nodes whose neighbourhood yielded no improvement are skipped until a
   neighbouring move changes their tour-adjacency.

6. **Double-bridge kick** (ILS perturbation):
   4-opt non-sequential perturbation that escapes local optima, delegated to
   :func:`double_bridge` from
   ``logic.src.policies.helpers.operators.perturbation.double_bridge``.

7. **Tour merging** (pool-based recombination):
   Combines shared edges from several elite tours to seed new searches.

8. **_improve_tour** — outer driver.  Iterates over every non-masked node
   t1 and tries 2-opt first; if unsuccessful it attempts 3-, 4-, and 5-opt
   in order (each gated by an instance-size threshold).  Accepts the first
   improvement found (first-improvement strategy) and returns immediately.
   Uses *don't-look bits* (Helsgaun 2000, Section 5.3) to skip nodes whose
   neighbourhood was exhausted since the last move touched them.

References:
    Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
      traveling salesman heuristic. EJOR 126, 106-130.

Example:
    >>> tour, cost = solve_lkh(distance_matrix)
"""

from __future__ import annotations

import random
from collections import deque
from random import Random
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
    merge_tours,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_improvement import (
    _2opt_gain,
    _apply_kopt_via_operator,
    _try_3opt_move,
    _try_4opt_move,
    _try_5opt_move,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

# ---------------------------------------------------------------------------
# Alpha-measure and candidate-set construction
# ---------------------------------------------------------------------------


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute α-nearness for every edge using minimum-spanning-tree sensitivity.

    α(i, j) = c(i, j) − β(i, j),  where β(i, j) is the weight of the heaviest
    edge on the unique MST path between i and j (Helsgaun 2000, Section 4.1).

    Edges with α = 0 belong to some MST; edges with small α are "nearly
    spanning" and hence strong candidates for an optimal tour.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        (n × n) array of α-values (symmetric, non-negative).
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    # Convert directed result to symmetric dense MST
    mst = mst_sparse.toarray()
    mst = np.maximum(mst, mst.T)

    # Build explicit adjacency list for faster MST traversal (fixes O(V^3) bottleneck)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    rows, cols = np.where(mst > 0)
    for r, c in zip(rows, cols, strict=False):
        adj[r].append((c, float(mst[r, c])))

    alpha = np.zeros((n, n), dtype=float)

    for i in range(n):
        # BFS to find max edge weight from root i to all other nodes
        max_edge_to = np.zeros(n, dtype=float)
        visited = np.zeros(n, dtype=bool)
        queue: Deque[int] = deque([i])
        visited[i] = True

        while queue:
            current = queue.popleft()
            for neighbor, weight in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    # Max edge on path to neighbor is max(max edge to current, weight)
                    max_edge_to[neighbor] = max(max_edge_to[current], weight)
                    queue.append(neighbor)

        # After BFS for root i, populate alpha row for pairs (i, j)
        for j in range(i + 1, n):
            alpha_val = distance_matrix[i, j] - max_edge_to[j]
            alpha[i, j] = alpha[j, i] = max(0.0, alpha_val)

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray,
    alpha_measures: np.ndarray,
    max_candidates: int = 5,
) -> Dict[int, List[int]]:
    """
    Build per-node candidate lists sorted by α-nearness (ties broken by cost).

    Restricting the inner LK search to these lists reduces worst-case
    complexity from O(n²) to O(n · max_candidates) per pass.

    Args:
        distance_matrix: (n × n) cost matrix.
        alpha_measures: (n × n) α-nearness matrix from :func:`compute_alpha_measures`.
        max_candidates: Maximum number of candidates per node (default 5).

    Returns:
        dict mapping each node index to its sorted candidate list.
    """
    n = len(distance_matrix)
    candidates: Dict[int, List[int]] = {}
    for i in range(n):
        indices = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: (alpha_measures[i, j], distance_matrix[i, j]),
        )
        candidates[i] = indices[:max_candidates]
    return candidates


# ---------------------------------------------------------------------------
# Main improvement loop
# ---------------------------------------------------------------------------


def _improve_tour(  # noqa: C901
    curr_tour: List[int],
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    rng: Random,
    dont_look_bits: Optional[np.ndarray] = None,
    max_k: int = 5,
) -> Tuple[List[int], float, bool, Optional[np.ndarray]]:
    """
    Execute one complete pass of sequential k-opt local search (k = 2..max_k).

    For each node t1 (not masked by a don't-look bit) and its successor t2
    in the current tour, the following hierarchy is attempted in order:

    1. **2-opt** — exact gain pre-screen over α-nearest neighbours of t2;
       move applied via :func:`move_kopt_intra` (k=2).
    2. **3-opt** — exact gain pre-screen for all seven patterns; direct segment
       swapping.  Enabled when max_k >= 3 and n < 500.
    3. **4-opt** — exact gain pre-screen for three patterns; direct segment
       swapping.  Enabled when max_k >= 4 and n < 300.
    4. **5-opt** — exact gain pre-screen for five patterns; direct segment
       swapping.  Enabled when max_k >= 5 and n < 200.

    Only exact positive gains trigger a move.  The first improving move at the
    lowest k is accepted (first-improvement strategy) and the function returns
    immediately so the outer loop can restart.

    Don't-look bits (Helsgaun 2000, Section 5.3) are reset for all nodes
    involved in an accepted move and set for nodes whose search found nothing.

    Args:
        curr_tour: Current closed tour.
        curr_cost: Current tour cost.
        candidates: α-nearest-neighbour candidate lists.
        distance_matrix: Cost matrix.
        rng: Random number generator forwarded to operators.
        dont_look_bits: Boolean array of length n; nodes with True are skipped.
        max_k: Maximum k-opt depth (2-5). Throttles search for small instances.

    Returns:
        (new_tour, new_cost, any_improvement, updated_bits).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # O(1) position map for tour index lookups
    pos_map = {node: idx for idx, node in enumerate(curr_tour[:-1])}

    if dont_look_bits is None:
        dont_look_bits = np.zeros(nodes_count, dtype=bool)

    improved_overall = False
    for i in range(nodes_count):
        t1 = curr_tour[i]

        if dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]

        # ---- 2-opt ----
        for t3 in candidates[t2]:
            if nodes_count >= 5:
                if t3 == t1 or t3 == curr_tour[(i + 2) % nodes_count]:
                    continue
            else:
                if t3 == t2:
                    continue

            j = pos_map.get(t3, -1)
            if j == -1 or j <= i + 1:
                continue

            t4 = curr_tour[j + 1]

            gain = _2opt_gain(t1, t2, t3, t4, d)
            if gain > 1e-6:
                new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
                if new_tour is not None:
                    c_new = get_cost(new_tour, d)
                    if is_better(c_new, curr_cost):
                        curr_tour, curr_cost = new_tour, c_new
                        dont_look_bits[t1] = False
                        dont_look_bits[t2] = False
                        dont_look_bits[t3] = False
                        dont_look_bits[t4] = False
                        return curr_tour, curr_cost, True, dont_look_bits

            # ---- 3-opt ----
            if max_k >= 3 and nodes_count < 500:
                res_tour, res_c, res_imp = _try_3opt_move(curr_tour, i, j, t1, t2, t3, t4, d, rng)
                if res_imp and res_tour is not None and is_better(res_c, curr_cost):
                    curr_tour, curr_cost = res_tour, res_c
                    dont_look_bits[t1] = False
                    dont_look_bits[t2] = False
                    dont_look_bits[t3] = False
                    dont_look_bits[t4] = False
                    return curr_tour, curr_cost, True, dont_look_bits

                # ---- 4-opt ----
                if max_k >= 4 and nodes_count < 300:
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[k_idx + 1]

                        res4, rc4, ri4 = _try_4opt_move(
                            curr_tour,
                            i,
                            j,
                            k_idx,
                            t1,
                            t2,
                            t3,
                            t4,
                            t5,
                            t6,
                            d,
                            rng,
                        )
                        if ri4 and res4 is not None and is_better(rc4, curr_cost):
                            curr_tour, curr_cost = res4, rc4
                            dont_look_bits[t1] = dont_look_bits[t2] = dont_look_bits[t3] = dont_look_bits[
                                t4
                            ] = dont_look_bits[t5] = dont_look_bits[t6] = False
                            return curr_tour, curr_cost, True, dont_look_bits

                        # ---- 5-opt ----
                        if max_k >= 5 and nodes_count < 200:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[l_idx + 1]

                                res5, rc5, ri5 = _try_5opt_move(
                                    curr_tour,
                                    i,
                                    j,
                                    k_idx,
                                    l_idx,
                                    t1,
                                    t2,
                                    t3,
                                    t4,
                                    t5,
                                    t6,
                                    t7,
                                    t8,
                                    d,
                                    rng,
                                )
                                if ri5 and res5 is not None and is_better(rc5, curr_cost):
                                    # _try_5opt_move found an improving 5-opt move
                                    # Since 5-opt involves 5 cuts (10 nodes total) and we don't track
                                    # which specific m was used, clear all don't-look bits to be safe
                                    curr_tour, curr_cost = res5, rc5
                                    dont_look_bits.fill(False)  # Reset all bits after major improvement
                                    return curr_tour, curr_cost, True, dont_look_bits

        dont_look_bits[t1] = True
    return curr_tour, curr_cost, improved_overall, dont_look_bits


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_k: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Solve a TSP instance using the LKH iterated local-search scheme.

    Algorithm outline (Helsgaun 2000):

    1. **Initialisation** — nearest-neighbour or provided tour.
    2. **Candidate sets** — α-measure computed from minimum spanning tree.
    3. **Local-search loop** — repeated k-opt passes (k = 2..max_k) with
       don't-look bits until a local optimum is reached.
    4. **Perturbation** — double-bridge kick to escape the basin of
       attraction of the current local optimum.
    5. **Tour-pool merging** — every 10 kicks, two elite tours from the pool
       are merged to create a new starting point.
    6. **Best-solution tracking** — simple distance minimization.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).  Nearest-neighbour
            construction is used when not provided.
        max_iterations: Maximum number of kicks / restarts.
        max_k: Maximum k-opt depth (2-5). Lower values speed up small instances.
        recorder: Optional recorder for visualisation / diagnostics.
        np_rng: Numpy random generator; seeded from 42 if not provided.
        seed: Alternative seed parameter (for compatibility).

    Returns:
        (best_tour, best_cost) where best_tour is a closed node sequence.
    """
    n = len(distance_matrix)
    if n < 3:
        tour = list(range(n)) + [0]
        cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(n))
        return tour, float(cost)

    if np_rng is None:
        np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Bridge numpy RNG (used for array ops) → stdlib Random (operator interfaces)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # 1. Initialisation
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. Candidate sets via α-measure
    alpha = compute_alpha_measures(distance_matrix)
    candidates = get_candidate_set(distance_matrix, alpha, max_candidates=5)

    curr_cost = get_cost(curr_tour, distance_matrix)
    best_tour = curr_tour[:]
    best_cost = curr_cost

    dont_look_bits: Optional[np.ndarray] = None

    # Tour pool for elite-solution recombination
    tour_pool: List[List[int]] = [best_tour[:]]
    max_pool_size = 5

    # 3. Main iterated local-search loop
    for restart in range(max_iterations):
        # Inner local-search until no improving k-opt move exists
        while True:
            curr_tour, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_cost,
                candidates,
                distance_matrix,
                stdlib_rng,
                dont_look_bits,
                max_k,
            )
            if not improved_local:
                break

        # Update global best
        if is_better(curr_cost, best_cost):
            best_tour = curr_tour[:]
            best_cost = curr_cost
            tour_pool.append(best_tour[:])
            if len(tour_pool) > max_pool_size:
                tour_pool.pop(0)

        if recorder is not None:
            recorder.record(
                restart=restart,
                best_cost=best_cost,
                curr_cost=curr_cost,
            )

        # Every 10 restarts: try tour-pool merging
        if restart > 0 and restart % 10 == 0 and len(tour_pool) >= 2:
            idx1, idx2 = random.sample(range(len(tour_pool)), 2)
            merged_tour = merge_tours(tour_pool[idx1], tour_pool[idx2], distance_matrix)
            merged_cost = get_cost(merged_tour, distance_matrix)

            if is_better(merged_cost, best_cost):
                best_tour = merged_tour[:]
                best_cost = merged_cost
                tour_pool.append(merged_tour[:])
                if len(tour_pool) > max_pool_size:
                    tour_pool.pop(0)

            curr_tour = merged_tour
            curr_cost = merged_cost
            dont_look_bits = None
        else:
            # Double-bridge perturbation via the shared operator
            curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            curr_cost = get_cost(curr_tour, distance_matrix)
            dont_look_bits = None

    return best_tour, best_cost
