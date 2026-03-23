"""
POPMUSIC Candidate Generation Module.

Implements the POPMUSIC (Partial Optimization Metaheuristic Under Special
Intensification Conditions) candidate-set generation strategy for the LKH-3
solver, designed to scale to large instances (N > 1000) where global MST-based
α-measure computation is prohibitively expensive.

Algorithm outline (Taillard & Voss 2002, adapted for LKH-3):

1. Start from an initial rough tour (e.g., nearest-neighbour).
2. Decompose the tour into overlapping sub-paths of size *r*.
3. Optimize each sub-path independently using 2-opt local search.
4. Repeat the decomposition from *K* independent random starting tours.
5. Collect the union of all edges from all optimized sub-paths
   to form a sparse candidate graph.

The resulting candidate dict has the same interface as
:func:`get_candidate_set` from the main LKH module but is computed in
O(K · n · r²) instead of O(n²).

References:
    Taillard, É. D. & Voss, S. (2002). POPMUSIC — Partial optimization
      metaheuristic under special intensification conditions. OR,
      Springer, 613-629.
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for
      constrained traveling salesman and vehicle routing problems.

Example:
    >>> from logic.src.policies.lin_kernighan_helsgaun_three._popmusic import (
    ...     popmusic_candidates
    ... )
    >>> candidates = popmusic_candidates(dist, tour, subpath_size=50, n_runs=5)
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sub-path decomposition
# ---------------------------------------------------------------------------


def decompose_tour(
    tour: List[int],
    subpath_size: int,
    overlap: int = 5,
) -> List[List[int]]:
    """Decompose a closed tour into overlapping sub-paths.

    Each sub-path contains `subpath_size` consecutive nodes from the tour.
    Successive sub-paths overlap by `overlap` nodes so that edges near
    sub-path boundaries can still participate in local optimization.

    Args:
        tour: Closed tour (first node repeated at end).
        subpath_size: Number of nodes per sub-path (r).
        overlap: Number of overlapping nodes between successive sub-paths.

    Returns:
        List of sub-path node lists.
    """
    # Work with the open tour (no closing duplicate)
    open_tour = tour[:-1] if len(tour) > 1 and tour[0] == tour[-1] else tour[:]
    n = len(open_tour)

    if n <= subpath_size:
        return [open_tour[:]]

    step = max(1, subpath_size - overlap)
    subpaths: List[List[int]] = []

    for start in range(0, n, step):
        end = start + subpath_size
        if end <= n:
            subpaths.append(open_tour[start:end])
        else:
            # Wrap around: take from end of tour + beginning
            wrap = open_tour[start:] + open_tour[: end - n]
            subpaths.append(wrap)
            break  # One wrap-around sub-path is sufficient

    return subpaths


# ---------------------------------------------------------------------------
# Sub-path optimization (fast 2-opt)
# ---------------------------------------------------------------------------


def optimize_subpath(
    subpath: List[int],
    distance_matrix: np.ndarray,
    max_trials: int = 50,
) -> List[int]:
    """Optimize a sub-path using repeated 2-opt local search.

    Performs a simple first-improvement 2-opt sweep for up to `max_trials`
    passes.  This is intentionally lightweight — the goal is to discover
    good edges, not to find optimal sub-tours.

    Args:
        subpath: Node sequence to optimize (open, no wrap).
        distance_matrix: Full (N × N) cost matrix.
        max_trials: Maximum number of 2-opt improvement passes.

    Returns:
        The optimized sub-path (same nodes, potentially reordered).
    """
    path = subpath[:]
    n = len(path)
    if n < 4:
        return path

    d = distance_matrix
    improved = True
    trials = 0

    while improved and trials < max_trials:
        improved = False
        trials += 1
        for i in range(n - 2):
            for j in range(i + 2, n):
                # Skip if j is the last and i is the first (would just reverse entire path)
                if i == 0 and j == n - 1:
                    continue

                # 2-opt gain: remove (i, i+1) and (j, j+1 mod n),
                # reconnect as (i, j) and (i+1, j+1 mod n)
                a, b = path[i], path[i + 1]
                c = path[j]
                e = path[(j + 1) % n] if j + 1 < n else path[0]

                gain = d[a, b] + d[c, e] - d[a, c] - d[b, e]
                if gain > 1e-6:
                    # Reverse the segment between i+1 and j
                    path[i + 1 : j + 1] = path[i + 1 : j + 1][::-1]
                    improved = True

    return path


# ---------------------------------------------------------------------------
# Edge collection helpers
# ---------------------------------------------------------------------------


def _collect_edges(path: List[int]) -> Set[Tuple[int, int]]:
    """Collect all undirected edges from a node sequence.

    Args:
        path: Node sequence (open or closed).

    Returns:
        Set of (min_node, max_node) tuples.
    """
    edges: Set[Tuple[int, int]] = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edges.add((min(a, b), max(a, b)))
    return edges


# ---------------------------------------------------------------------------
# Main POPMUSIC candidate generation
# ---------------------------------------------------------------------------


def popmusic_candidates(
    distance_matrix: np.ndarray,
    initial_tour: List[int],
    subpath_size: int = 50,
    n_runs: int = 5,
    max_trials: int = 50,
    max_candidates: int = 5,
    np_rng: Optional[np.random.Generator] = None,
) -> Dict[int, List[int]]:
    """Generate candidate sets via POPMUSIC for large instances.

    Runs *n_runs* independent decompose-optimize cycles on randomly
    shuffled copies of the initial tour, collects the union of all
    edges from optimized sub-paths, and builds per-node candidate lists
    sorted by distance (limited to `max_candidates` per node).

    For small instances (N ≤ subpath_size), falls back to simple
    nearest-neighbour candidates.

    Args:
        distance_matrix: (N × N) symmetric cost matrix.
        initial_tour: A closed starting tour.
        subpath_size: Number of nodes per sub-path (r).
        n_runs: Number of independent POPMUSIC runs (K).
        max_trials: Maximum 2-opt passes per sub-path optimization.
        max_candidates: Maximum candidates per node in the result dict.
        np_rng: NumPy random generator for reproducibility.

    Returns:
        dict mapping each node index to its sorted candidate list.
    """
    n = len(distance_matrix)

    if np_rng is None:
        np_rng = np.random.default_rng(42)

    # --- Collect edges from K independent POPMUSIC runs ---
    all_edges: Set[Tuple[int, int]] = set()

    for _ in range(n_runs):
        # Create a randomized tour by shuffling the non-depot nodes
        open_tour = (
            initial_tour[:-1] if (len(initial_tour) > 1 and initial_tour[0] == initial_tour[-1]) else initial_tour[:]
        )

        # Shuffle the order for diversity (keep depot at start)
        if open_tour[0] == 0:
            interior = open_tour[1:]
            np_rng.shuffle(interior)
            shuffled_tour = [0] + list(interior) + [0]
        else:
            arr = np.array(open_tour)
            np_rng.shuffle(arr)
            shuffled_tour = list(arr) + [int(arr[0])]

        # Decompose and optimize
        subpaths = decompose_tour(shuffled_tour, subpath_size)
        for sp in subpaths:
            optimized = optimize_subpath(sp, distance_matrix, max_trials)
            all_edges |= _collect_edges(optimized)

    # Also collect edges from the original tour
    all_edges |= _collect_edges(initial_tour)

    # --- Build per-node adjacency from collected edges ---
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in all_edges:
        if 0 <= a < n and 0 <= b < n:
            adjacency[a].add(b)
            adjacency[b].add(a)

    # --- Build candidate lists, sorted by distance ---
    candidates: Dict[int, List[int]] = {}
    for node in range(n):
        # Union of POPMUSIC neighbors + nearest-by-distance fallback
        neighbors = adjacency[node]

        # If POPMUSIC didn't find enough neighbors, add nearest-by-distance
        if len(neighbors) < max_candidates:
            all_nodes = sorted(
                [j for j in range(n) if j != node],
                key=lambda j: distance_matrix[node, j],
            )
            for j in all_nodes:
                neighbors.add(j)
                if len(neighbors) >= max_candidates * 2:
                    break

        # Sort by distance and trim
        sorted_neighbors = sorted(
            neighbors,
            key=lambda j: distance_matrix[node, j],
        )
        candidates[node] = sorted_neighbors[:max_candidates]

    return candidates
