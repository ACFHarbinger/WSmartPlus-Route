"""
LKH-3 inspired Heuristic Module.

This module implements a version of the Lin-Kernighan-Helsgaun heuristic inspired by LKH-3.
Key features include Alpha-measure pruning, lexicographical optimization (Penalty, Cost),
and Iterated Local Search with 2-opt and 3-opt moves.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _find_mst_path_max(mst_adj: np.ndarray, start: int, end: int, n: int) -> float:
    """
    Find maximum edge weight on unique path in MST between start and end nodes.

    Uses BFS to find the path and track maximum edge weight encountered.
    """
    if start == end:
        return 0.0

    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    queue = [start]
    visited[start] = True

    # BFS to find path
    while queue:
        current = queue.pop(0)
        if current == end:
            break

        for neighbor in range(n):
            if not visited[neighbor] and (mst_adj[current, neighbor] > 0 or mst_adj[neighbor, current] > 0):
                visited[neighbor] = True
                parent[neighbor] = current
                queue.append(neighbor)

    # Backtrack to find max edge on path
    if parent[end] == -1:
        return 0.0  # No path found

    max_edge = 0.0
    current = end
    while parent[current] != -1:
        prev = parent[current]
        edge_weight = max(mst_adj[prev, current], mst_adj[current, prev])
        max_edge = max(max_edge, edge_weight)
        current = prev

    return max_edge


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Alpha-measures for edge pruning based on MST.

    alpha(i,j) = c(i,j) - (max edge weight on the unique path in MST between i and j).

    This is the proper LKH-3 definition. Edges with low alpha values are good candidates
    for inclusion in tours.
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst = mst_sparse.toarray()

    # Compute alpha for each edge
    alpha = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            # Find max edge on MST path from i to j
            max_mst_edge = _find_mst_path_max(mst, i, j, n)
            # Alpha = actual distance - max MST path edge
            alpha_val = distance_matrix[i, j] - max_mst_edge
            alpha[i, j] = alpha_val
            alpha[j, i] = alpha_val

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray, alpha_measures: np.ndarray, max_candidates: int = 5
) -> Dict[int, List[int]]:
    """Generate candidate sets of edges for each node based on Alpha-measures."""
    n = len(distance_matrix)
    candidates = {}
    for i in range(n):
        # Sort by alpha, then by distance
        # We can implement a composite sort key if needed, but alpha is primary.
        indices = np.argsort(alpha_measures[i])
        # Filter self
        valid_indices = [int(idx) for idx in indices if idx != i]
        # Strict limit
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
            current_load += waste[node]
            if current_load > capacity + 1e-6:
                penalty += current_load - capacity
    return penalty


def get_score(
    tour: List[int],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[float, float]:
    """
    Calculate total penalty and cost for a given tour.

    Returns:
        tuple: (penalty, cost)
    """
    n = len(tour) - 1  # items in tour (excluding duplicate end)
    # Assumes tour is closed [0...0]

    # Cost
    c = 0.0
    for i in range(n):
        c += distance_matrix[tour[i], tour[i + 1]]

    pen = calculate_penalty(tour, waste, capacity)
    return pen, c


def is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """Lexicographical comparison: Penalty first, then Cost."""
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def apply_2opt_move(tour: List[int], i: int, j: int) -> List[int]:
    """
    Apply 2-opt move: reverse segment between i+1 and j.
    Indices refer to positions in the open tour (0 to n-1).
    """
    new_tour = tour[:]
    # Python slicing is [start : end] exclusive.
    # Reverse elements from i+1 up to j (inclusive)
    new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
    return new_tour


def apply_3opt_move(tour: List[int], i: int, j: int, k: int, case: int) -> List[int]:
    """
    Apply 3-opt move with all reconnection cases.

    Given a tour with edges (i,i+1), (j,j+1), (k,k+1) to be removed,
    there are 7 non-trivial ways to reconnect the 3 segments:
    - Segment A: tour[0:i+1]
    - Segment B: tour[i+1:j+1]
    - Segment C: tour[j+1:k+1]
    - Segment D: tour[k+1:]

    Cases (following Helsgaun's LKH-3 implementation):
    0: A-B'-C'-D (reverse both B and C) - double reversal
    1: A-B-C'-D (reverse only C)
    2: A-B'-C-D (reverse only B)
    3: A-C-B-D (swap B and C, no reversal)
    4: A-C-B'-D (swap B and C, reverse B)
    5: A-C'-B-D (swap B and C, reverse C)
    6: A-C'-B'-D (swap B and C, reverse both)

    where B' denotes reversed B, C' denotes reversed C.
    """
    new_tour = tour[:]

    # Extract segments
    A = tour[0 : i + 1]
    B = tour[i + 1 : j + 1]
    C = tour[j + 1 : k + 1]
    D = tour[k + 1 :]

    if case == 0:
        # A-B'-C'-D: reverse both B and C
        new_tour = A + B[::-1] + C[::-1] + D
    elif case == 1:
        # A-B-C'-D: reverse only C
        new_tour = A + B + C[::-1] + D
    elif case == 2:
        # A-B'-C-D: reverse only B
        new_tour = A + B[::-1] + C + D
    elif case == 3:
        # A-C-B-D: swap B and C, no reversal
        new_tour = A + C + B + D
    elif case == 4:
        # A-C-B'-D: swap B and C, reverse B
        new_tour = A + C + B[::-1] + D
    elif case == 5:
        # A-C'-B-D: swap B and C, reverse C
        new_tour = A + C[::-1] + B + D
    elif case == 6:
        # A-C'-B'-D: swap B and C, reverse both
        new_tour = A + C[::-1] + B[::-1] + D
    else:
        # Invalid case, return original
        new_tour = tour[:]

    return new_tour


def apply_4opt_move(tour: List[int], i: int, j: int, k: int, l: int, case: int) -> List[int]:
    """
    Apply 4-opt move with various reconnection cases.

    Given a tour with edges (i,i+1), (j,j+1), (k,k+1), (l,l+1) to be removed,
    reconnect the 4 segments in different ways.

    Segments:
    - A: tour[0:i+1]
    - B: tour[i+1:j+1]
    - C: tour[j+1:k+1]
    - D: tour[k+1:l+1]
    - E: tour[l+1:]

    This implements common 4-opt reconnection patterns.
    """
    A = tour[0 : i + 1]
    B = tour[i + 1 : j + 1]
    C = tour[j + 1 : k + 1]
    D = tour[k + 1 : l + 1]
    E = tour[l + 1 :]

    # Implement some common 4-opt cases
    if case == 0:
        # Double bridge (as used in perturbation)
        new_tour = A + C + B + D + E
    elif case == 1:
        # Reverse B and D
        new_tour = A + B[::-1] + C + D[::-1] + E
    elif case == 2:
        # Swap B and D
        new_tour = A + D + C + B + E
    else:
        # Default: return original
        new_tour = tour[:]

    return new_tour


def apply_5opt_move(tour: List[int], i: int, j: int, k: int, l: int, m: int, case: int) -> List[int]:
    """
    Apply 5-opt move with various reconnection cases.

    Given a tour with edges (i,i+1), (j,j+1), (k,k+1), (l,l+1), (m,m+1) to be removed,
    reconnect the 5 segments.

    Segments:
    - A: tour[0:i+1]
    - B: tour[i+1:j+1]
    - C: tour[j+1:k+1]
    - D: tour[k+1:l+1]
    - E: tour[l+1:m+1]
    - F: tour[m+1:]

    Following Helsgaun (2000), any 5-opt move can be represented as a sequence
    of at most five 2-opt moves, but we implement direct reconnection for efficiency.
    """
    A = tour[0 : i + 1]
    B = tour[i + 1 : j + 1]
    C = tour[j + 1 : k + 1]
    D = tour[k + 1 : l + 1]
    E = tour[l + 1 : m + 1]
    F = tour[m + 1 :]

    # Implement some common 5-opt reconnection patterns
    # These are heuristic choices based on what typically improves tours
    if case == 0:
        # Reverse alternating segments
        new_tour = A + B[::-1] + C + D[::-1] + E + F
    elif case == 1:
        # Swap and reverse
        new_tour = A + C + B[::-1] + D + E + F
    elif case == 2:
        # Complex rearrangement
        new_tour = A + B + D + C[::-1] + E + F
    else:
        # Default: return original
        new_tour = tour[:]

    return new_tour


def merge_tours(tour1: List[int], tour2: List[int], distance_matrix: np.ndarray) -> List[int]:
    """
    Merge two tours to create a new tour combining common edges.

    Following Helsgaun's tour merging concept: extract edges that appear
    in both tours and use them as a basis for constructing a new tour.

    Args:
        tour1: First tour
        tour2: Second tour
        distance_matrix: Distance matrix

    Returns:
        Merged tour
    """
    n = len(tour1) - 1

    # Find common edges between the two tours
    edges1 = set()
    edges2 = set()

    for i in range(len(tour1) - 1):
        # Add edge in both directions (undirected)
        a, b = tour1[i], tour1[i + 1]
        edges1.add((min(a, b), max(a, b)))

    for i in range(len(tour2) - 1):
        a, b = tour2[i], tour2[i + 1]
        edges2.add((min(a, b), max(a, b)))

    # Common edges
    common_edges = edges1 & edges2

    if not common_edges:
        # No common edges - return the better tour
        cost1 = sum(distance_matrix[tour1[i], tour1[i + 1]] for i in range(len(tour1) - 1))
        cost2 = sum(distance_matrix[tour2[i], tour2[i + 1]] for i in range(len(tour2) - 1))
        return tour1 if cost1 <= cost2 else tour2

    # Build adjacency list from common edges
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in common_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Try to construct a tour using common edges as much as possible
    # Start from depot (node 0)
    visited = [False] * n
    merged_tour = [0]
    visited[0] = True
    current = 0

    while len(merged_tour) < n:
        # Try to follow common edges first
        next_node = None

        # Check neighbors in common edges
        for neighbor in adj[current]:
            if not visited[neighbor]:
                next_node = neighbor
                break

        # If no common edge available, pick nearest unvisited node
        if next_node is None:
            min_dist = float("inf")
            for node in range(n):
                if not visited[node]:
                    dist = distance_matrix[current, node]
                    if dist < min_dist:
                        min_dist = dist
                        next_node = node

        if next_node is None:
            break

        merged_tour.append(next_node)
        visited[next_node] = True
        current = next_node

    # Close the tour
    merged_tour.append(0)

    return merged_tour


def double_bridge_kick(tour: List[int], np_rng: np.random.Generator) -> List[int]:
    """
    Apply a Double Bridge kick (random 4-opt move).
    Breaks 4 edges and reconnects to create a major perturbation.
    """
    n = len(tour) - 1  # active nodes
    if n < 8:
        return tour  # Too small

    pos = sorted(np_rng.choice(range(1, n - 1), 4, replace=False))
    a, b, c, d = pos
    # Segments: [0..a], [a+1..b], [b+1..c], [c+1..d], [d+1..end]
    # Reconnect: [0..a] -> [c+1..d] -> [b+1..c] -> [a+1..b] -> [d+1..end]

    new_tour = tour[: a + 1] + tour[c + 1 : d + 1] + tour[b + 1 : c + 1] + tour[a + 1 : b + 1] + tour[d + 1 :]
    return new_tour


def _initialize_tour(distance_matrix: np.ndarray, initial_tour: Optional[List[int]]) -> List[int]:
    """Initialize tour using nearest neighbor if not provided."""
    n = len(distance_matrix)
    if initial_tour is None:
        # Nearest Neighbor Construction
        curr = 0
        path = [0]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
            path.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        path.append(0)
        return path
    else:
        curr_tour = initial_tour[:]
        if curr_tour[0] != curr_tour[-1]:
            curr_tour.append(curr_tour[0])
        return curr_tour


def _try_2opt_move(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """Attempt 2-opt moves for edge (t1, t2)."""
    nodes_count = len(curr_tour) - 1
    # Check candidates for t2
    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        if t3 == curr_tour[(i + 2) % nodes_count]:
            continue  # skip next edge

        # 2-opt Attempt
        try:
            j = curr_tour.index(t3)
        except ValueError:
            continue

        # Case A: j > i
        if j <= i + 1:
            continue

        t4 = curr_tour[j + 1]

        # Gain check roughly
        gain = (distance_matrix[t1, t2] + distance_matrix[t3, t4]) - (distance_matrix[t1, t3] + distance_matrix[t2, t4])

        if gain > 1e-6:
            new_tour = apply_2opt_move(curr_tour, i, j)
            p_new, c_new = get_score(new_tour, distance_matrix, waste, capacity)
            return new_tour, p_new, c_new, True, j

    return None, 0.0, 0.0, False, -1


def _try_3opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Attempt 3-opt moves with all reconnection cases.

    Following Helsgaun (2000), tries multiple reconnection patterns
    to find improving moves.
    """
    nodes_count = len(curr_tour) - 1
    for k in range(j + 2, nodes_count):
        t5 = curr_tour[k]
        t6 = curr_tour[k + 1]

        # Try different 3-opt reconnection cases (0-6)
        for case in range(7):
            # Compute approximate gain for this case
            # Case 0: double reversal (original implementation)
            if case == 0:
                gain = (distance_matrix[t1, t2] + distance_matrix[t3, t4] + distance_matrix[t5, t6]) - (
                    distance_matrix[t1, t3] + distance_matrix[t2, t5] + distance_matrix[t4, t6]
                )
            else:
                # For other cases, use heuristic: if any pair looks promising, try it
                # This is a simplification - full LKH would compute exact gains
                gain = 0.1  # Allow exploration

            if gain > -1e-6:
                new_3opt = apply_3opt_move(curr_tour, i, j, k, case)
                p3, c3 = get_score(new_3opt, distance_matrix, waste, capacity)
                # Check if this is actually an improvement
                curr_p, curr_c = get_score(curr_tour, distance_matrix, waste, capacity)
                if is_better(p3, c3, curr_p, curr_c):
                    return new_3opt, p3, c3, True

    return None, 0.0, 0.0, False


def _try_4opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    _t1: int,  # Reserved for future exact gain calculations
    _t2: int,
    _t3: int,
    _t4: int,
    _t5: int,
    _t6: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Attempt 4-opt moves.

    Following Helsgaun (2000), explores 4-opt reconnections.
    Node parameters (t1-t6) are reserved for future exact gain calculations.
    """
    nodes_count = len(curr_tour) - 1
    curr_p, curr_c = get_score(curr_tour, distance_matrix, waste, capacity)

    for l in range(k + 2, nodes_count):
        _t7 = curr_tour[l]
        _t8 = curr_tour[l + 1]

        # Try different 4-opt reconnection cases
        for case in range(3):  # We have 3 cases in apply_4opt_move
            new_4opt = apply_4opt_move(curr_tour, i, j, k, l, case)
            p4, c4 = get_score(new_4opt, distance_matrix, waste, capacity)

            if is_better(p4, c4, curr_p, curr_c):
                return new_4opt, p4, c4, True

    return None, 0.0, 0.0, False


def _try_5opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
    _t1: int,  # Reserved for future exact gain calculations
    _t2: int,
    _t3: int,
    _t4: int,
    _t5: int,
    _t6: int,
    _t7: int,
    _t8: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Attempt 5-opt moves - the key innovation of Helsgaun (2000).

    Following Section 4.3: "the basic move is now a sequential 5-opt move"
    Node parameters (t1-t8) are reserved for future exact gain calculations.
    """
    nodes_count = len(curr_tour) - 1
    curr_p, curr_c = get_score(curr_tour, distance_matrix, waste, capacity)

    for m in range(l + 2, nodes_count):
        _t9 = curr_tour[m]
        _t10 = curr_tour[m + 1]

        # Try different 5-opt reconnection cases
        for case in range(3):  # We have 3 cases in apply_5opt_move
            new_5opt = apply_5opt_move(curr_tour, i, j, k, l, m, case)
            p5, c5 = get_score(new_5opt, distance_matrix, waste, capacity)

            if is_better(p5, c5, curr_p, curr_c):
                return new_5opt, p5, c5, True

    return None, 0.0, 0.0, False


def _improve_tour(  # noqa: C901 - Sequential k-opt search inherently complex
    curr_tour: List[int],
    curr_pen: float,
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    dont_look_bits: Optional[np.ndarray] = None,
) -> Tuple[List[int], float, float, bool, Optional[np.ndarray]]:
    """
    Run one pass of local search improvement with don't-look-bits optimization.

    Implements Helsgaun (2000) sequential k-opt search (k=2,3,4,5).
    Complexity is inherent to the algorithm's design.

    Returns:
        (new_tour, new_penalty, new_cost, improved, updated_dont_look_bits)
    """
    nodes_count = len(curr_tour) - 1

    # Initialize don't-look-bits if not provided
    if dont_look_bits is None:
        dont_look_bits = np.zeros(nodes_count, dtype=bool)

    improved_overall = False

    for i in range(nodes_count):
        t1 = curr_tour[i]

        # Skip nodes with don't-look-bit set (Section 5.3, refinement 3)
        if dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]
        found_improvement_for_t1 = False

        # 1. Try 2-opt
        # We need to iterate candidates manually to support 3-opt fallback
        # Re-implementing logic from original to support extraction
        for t3 in candidates[t2]:
            # For very small instances, allow all moves to find the best ordering
            if nodes_count >= 5:
                if t3 == t1 or t3 == curr_tour[(i + 2) % nodes_count]:
                    continue
            else:
                if t3 == t2:
                    continue
            try:
                j = curr_tour.index(t3)
            except ValueError:
                continue

            if j <= i + 1:
                continue

            t4 = curr_tour[j + 1]
            gain = (distance_matrix[t1, t2] + distance_matrix[t3, t4]) - (
                distance_matrix[t1, t3] + distance_matrix[t2, t4]
            )

            if gain > -1e-6:
                new_tour = apply_2opt_move(curr_tour, i, j)
                p_new, c_new = get_score(new_tour, distance_matrix, waste, capacity)
                if is_better(p_new, c_new, curr_pen, curr_cost):
                    # Improvement found! Reset don't-look-bits for affected nodes
                    # Nodes involved in the move: t1, t2, t3, t4
                    dont_look_bits[t1] = False
                    dont_look_bits[t2] = False
                    dont_look_bits[t3] = False
                    dont_look_bits[t4] = False

                    return new_tour, p_new, c_new, True, dont_look_bits

            # 2. Try 3-opt if 2-opt didn't improve
            # Helsgaun (2000) Section 4.3: sequential k-opt moves
            if len(distance_matrix) < 500:
                res_tour, res_p, res_c, res_imp = _try_3opt_move(
                    curr_tour,
                    i,
                    j,
                    t1,
                    t2,
                    t3,
                    t4,
                    distance_matrix,
                    waste,
                    capacity,
                )
                if res_imp and res_tour is not None and is_better(res_p, res_c, curr_pen, curr_cost):
                    # Improvement found via 3-opt! Reset don't-look-bits
                    dont_look_bits[t1] = False
                    dont_look_bits[t2] = False
                    dont_look_bits[t3] = False
                    dont_look_bits[t4] = False

                    return res_tour, res_p, res_c, True, dont_look_bits

                # 3. Try 4-opt if 3-opt didn't improve
                # Only for smaller instances to control complexity
                if len(distance_matrix) < 300:
                    # Find k for 4-opt
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[k_idx + 1]

                        res_tour_4, res_p_4, res_c_4, res_imp_4 = _try_4opt_move(
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
                            distance_matrix,
                            waste,
                            capacity,
                        )
                        if res_imp_4 and res_tour_4 is not None and is_better(res_p_4, res_c_4, curr_pen, curr_cost):
                            # Reset don't-look-bits for all affected nodes
                            dont_look_bits[t1] = False
                            dont_look_bits[t2] = False
                            dont_look_bits[t3] = False
                            dont_look_bits[t4] = False
                            dont_look_bits[t5] = False
                            dont_look_bits[t6] = False

                            return res_tour_4, res_p_4, res_c_4, True, dont_look_bits

                        # 4. Try 5-opt - the key innovation of Helsgaun (2000)
                        # Section 4.3: "the basic move is now a sequential 5-opt move"
                        # Only for very small instances due to computational cost
                        if len(distance_matrix) < 200:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[l_idx + 1]

                                res_tour_5, res_p_5, res_c_5, res_imp_5 = _try_5opt_move(
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
                                    distance_matrix,
                                    waste,
                                    capacity,
                                )
                                if (
                                    res_imp_5
                                    and res_tour_5 is not None
                                    and is_better(res_p_5, res_c_5, curr_pen, curr_cost)
                                ):
                                    # Reset don't-look-bits for all affected nodes
                                    dont_look_bits[t1] = False
                                    dont_look_bits[t2] = False
                                    dont_look_bits[t3] = False
                                    dont_look_bits[t4] = False
                                    dont_look_bits[t5] = False
                                    dont_look_bits[t6] = False
                                    dont_look_bits[t7] = False
                                    dont_look_bits[t8] = False

                                    return res_tour_5, res_p_5, res_c_5, True, dont_look_bits

        # No improvement found starting from node t1
        # Set don't-look-bit for t1
        if not found_improvement_for_t1:
            dont_look_bits[t1] = True

    return curr_tour, curr_pen, curr_cost, improved_overall, dont_look_bits


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    waste: Optional[np.ndarray] = None,
    capacity: Optional[float] = None,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], float]:
    """
    Solve TSP/VRP using refined Lin-Kernighan heuristics.

    Args:
        distance_matrix: NxN distance matrix.
        initial_tour: Optional initial tour.
        max_iterations: Max improvements/kicks.
        waste: Node weights.
        capacity: Vehicle capacity.
        np_rng: Numpy random number generator.

    Returns:
        (best_tour, best_cost)
    """
    n = len(distance_matrix)
    if n < 3:
        tour = list(range(n)) + [0]
        cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(n))
        return tour, float(cost)

    # 1. Initialization
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. Candidates
    alpha = compute_alpha_measures(distance_matrix)
    candidates = get_candidate_set(distance_matrix, alpha, max_candidates=5)

    curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity)
    best_tour, best_pen, best_cost = curr_tour[:], curr_pen, curr_cost

    # Initialize don't-look-bits for all nodes (Section 5.3, refinement 3)
    dont_look_bits: Optional[np.ndarray] = None

    # Tour pool for tour merging (Helsgaun's tour merging concept)
    tour_pool: List[List[int]] = [best_tour[:]]
    max_pool_size = 5

    # 3. Main Loop (Iterated Local Search using Kicks)
    for _restart in range(max_iterations):
        # Local Search Loop (2-opt / 3-opt / 4-opt / 5-opt until local optimal)
        while True:
            # delegated to helper with don't-look-bits
            curr_tour, curr_pen, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_pen,
                curr_cost,
                candidates,
                distance_matrix,
                waste,
                capacity,
                dont_look_bits,
            )
            if not improved_local:
                break

        # Update Global Best
        if is_better(curr_pen, curr_cost, best_pen, best_cost):
            best_tour, best_pen, best_cost = curr_tour[:], curr_pen, curr_cost

            # Add to tour pool for potential merging
            tour_pool.append(best_tour[:])
            if len(tour_pool) > max_pool_size:
                tour_pool.pop(0)  # Remove oldest tour

        if recorder is not None:
            recorder.record(restart=_restart, best_cost=best_cost, curr_cost=curr_cost, best_penalty=best_pen)

        # Occasionally try tour merging (every 10 iterations)
        if _restart > 0 and _restart % 10 == 0 and len(tour_pool) >= 2:
            # Merge two random tours from the pool
            idx1, idx2 = random.sample(range(len(tour_pool)), 2)
            merged_tour = merge_tours(tour_pool[idx1], tour_pool[idx2], distance_matrix)

            # Try to improve the merged tour
            merged_pen, merged_cost = get_score(merged_tour, distance_matrix, waste, capacity)

            if is_better(merged_pen, merged_cost, best_pen, best_cost):
                best_tour, best_pen, best_cost = merged_tour[:], merged_pen, merged_cost
                tour_pool.append(merged_tour[:])
                if len(tour_pool) > max_pool_size:
                    tour_pool.pop(0)

            # Use merged tour as starting point for next iteration
            curr_tour = merged_tour
            curr_pen, curr_cost = merged_pen, merged_cost
            dont_look_bits = None
        else:
            # Perturbation (Kick) - reset don't-look-bits after perturbation
            curr_tour = double_bridge_kick(best_tour, np_rng)
            curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity)
            # Reset don't-look-bits after perturbation since tour structure changed
            dont_look_bits = None

    return best_tour, best_cost
