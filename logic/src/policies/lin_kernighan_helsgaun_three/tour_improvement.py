"""
LKH-3 Tour Improvement Components.

Implements the sequential k-opt local-search loop (k = 2..5) used
in the Lin-Kernighan-Helsgaun heuristic (Helsgaun 2000, Sections 3–4).

These routines iterate over potential cut positions and use exact gain
pre-screening before delegating the actual segment swaps to the shared
:func:`move_kopt_intra` operator.

The high-level driver for these components is :func:`_improve_tour` in
``lin_kernighan_helsgaun.py``.

Architecture
------------
The improvement routines are granular:

1. **_apply_kopt_via_operator** — lowest-level bridge.  Wraps a tour in a
   :class:`TourAdapter`, calls ``move_kopt_intra(k=k)``, and returns the
   updated closed tour if the operator applied a move.

2. **_try_2opt_move** — 2-opt search over α-nearest neighbours of t2.
   Uses :func:`_2opt_gain` as a pre-screen so the operator is only called
   when the exact gain is positive.

3. **_try_3opt_move** — 3-opt search over all valid third cut positions.
   Uses :func:`_3opt_gains` (seven cases) as a pre-screen; cases 1 and 2
   (pure 2-opt sub-moves) are excluded to avoid duplicating the 2-opt phase.

4. **_try_4opt_move** — 4-opt search over all valid fourth cut positions.
   Uses :func:`_4opt_gains` (three cases) as a pre-screen.

5. **_try_5opt_move** — 5-opt search over all valid fifth cut positions.
   Uses :func:`_5opt_gains` (five cases) as a pre-screen. This is the
   "basic move" of Helsgaun (2000), Section 4.3.

Instance-size thresholds
------------------------
Higher-order moves scale super-linearly in the number of cut positions, so
they are restricted by route length:

- 3-opt: n < 500
- 4-opt: n < 300
- 5-opt: n < 200

Dependencies
------------
- :func:`get_score`, :func:`is_better` from ``._objective``
- :class:`TourAdapter` from ``._tour_adapter``
- :func:`_2opt_gain`, :func:`_3opt_gains`, :func:`_4opt_gains`,
  :func:`_5opt_gains` from ``._tour_construction``
- :func:`move_kopt_intra` from the shared intra-route operator package

Typical usage
-------------
>>> from logic.src.policies.other.operators.heuristics._tour_improvement import (
...     _2opt_gain,               # Exact gain for 2-opt
...     _apply_kopt_via_operator,  # Bridge to move_kopt_intra
...     _try_3opt_move,           # Sequential 3-opt search
...     _try_4opt_move,           # Sequential 4-opt search
...     _try_5opt_move,           # Sequential 5-opt search
... )
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
    get_score,
    is_better,
    penalty_delta,
)
from logic.src.policies.lin_kernighan_helsgaun_three.tour_adapter import TourAdapter
from logic.src.policies.lin_kernighan_helsgaun_three.tour_construction import (
    _2opt_gain,
    _3opt_gains,
    _4opt_gains,
    _5opt_gains,
    _should_accept_kopt_move,
)
from logic.src.policies.other.operators.intra_route.k_opt import move_kopt_intra

# ---------------------------------------------------------------------------
# k-opt move application via the shared operator
# ---------------------------------------------------------------------------


def _apply_kopt_via_operator(
    tour: List[int],
    p_u: int,
    p_v: int,
    k: int,
    distance_matrix: np.ndarray,
    rng: Random,
) -> Optional[List[int]]:
    """
    Attempt a k-opt move via :func:`move_kopt_intra` and return the new tour.

    Wraps the tour in a :class:`TourAdapter`, calls ``move_kopt_intra`` with
    the given cut positions, and returns the resulting closed tour only if the
    operator applied an improving move (i.e. ``_update_map`` was called).

    ``move_kopt_intra`` behaviour by k:

    - **k=2**: Checks ``delta * C < -1e-4`` (exact cost improvement).
    - **k=3**: Randomly samples a third cut and evaluates four reconnection
      patterns (g4..g7), applying the best improving one.
    - **k≥4**: Enumerates all permutations × orientations of the middle
      segments and applies the configuration with the greatest saving.

    Args:
        tour: Current closed tour.
        p_u: Position of the first cut node in the open route.
        p_v: Position of the second cut node in the open route.
        k: Number of edges to remove (2, 3, 4, or 5).
        distance_matrix: (n × n) cost matrix.
        rng: Random number generator (required for k ≥ 3).

    Returns:
        The new closed tour if an improvement was applied; ``None`` otherwise.
    """
    adapter = TourAdapter(tour, distance_matrix)
    route = adapter.routes[0]
    n = len(route)

    if p_u < 0 or p_v < 0 or p_u >= n or p_v >= n or p_u == p_v:
        return None

    u = route[p_u]
    v = route[p_v]

    applied = move_kopt_intra(
        adapter,
        u=u,
        v=v,
        r_u=0,
        p_u=p_u,
        r_v=0,
        p_v=p_v,
        k=k,
        rng=rng if k >= 3 else None,
    )

    if applied:
        return adapter.to_closed_tour()
    return None


# ---------------------------------------------------------------------------
# Inner search routines with gain pre-screening
# ---------------------------------------------------------------------------


def _try_2opt_move(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    Iterates over α-nearest neighbours of t2 as the candidate for t3.
    Uses lexicographic pre-screening to accept moves that either:
    1. Reduce capacity violations (ΔP < 0), OR
    2. Maintain feasibility (ΔP == 0) AND improve cost (ΔC > 0)

    This allows the search to transit through infeasible space by accepting
    penalty-reducing moves even if they increase routing cost.

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Get current tour score for penalty delta calculation
    p_curr, c_curr = get_score(curr_tour, d, waste, capacity, n_original)

    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        if t3 == curr_tour[(i + 2) % nodes_count]:
            continue

        try:
            j = curr_tour.index(t3)
        except ValueError:
            continue

        if j <= i + 1:
            continue

        t4 = curr_tour[j + 1]

        # Compute distance gain (ΔC)
        delta_c = _2opt_gain(t1, t2, t3, t4, d)

        # For VRP: Compute candidate new tour to evaluate penalty change
        if waste is not None and capacity is not None:
            # Create candidate tour for penalty evaluation
            candidate_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
            if candidate_tour is None:
                continue

            # Compute penalty delta (ΔP)
            delta_p = penalty_delta(curr_tour, candidate_tour, waste, capacity, n_original)

            # Lexicographic pre-screening
            if _should_accept_kopt_move(delta_p, delta_c):
                p_new, c_new = get_score(candidate_tour, d, waste, capacity, n_original)
                return candidate_tour, p_new, c_new, True, j
        else:
            # TSP mode: pure cost gain
            if delta_c > 1e-6:
                new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
                if new_tour is not None:
                    p_new, c_new = get_score(new_tour, d, waste, capacity, n_original)
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
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[object] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 3-opt move with O(1) lexicographic pre-screening.

    **CRITICAL FIX**: Moves _apply_kopt_via_operator() call AFTER lexicographic
    gate to avoid O(N^4) → O(N^3) complexity reduction.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator passed to the operator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for O(1) penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved).
        Returns (None, 0, 0, False) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[k_pos + 1]

        # Step 1: Compute distance gains (O(1))
        gains = _3opt_gains(t1, t2, t3, t4, t5, t6, d)
        # Cases 1 & 2 are pure 2-opt; skip to avoid duplication
        relevant_gains = [g for idx, g in enumerate(gains) if idx not in (1, 2)]
        best_gain = max(relevant_gains) if relevant_gains else 0.0

        # Step 2: O(1) penalty delta via load_state
        if load_state is not None and waste is not None and capacity is not None:
            from logic.src.policies.lin_kernighan_helsgaun_three.load_tracker import (
                calculate_penalty_delta_fast,
            )

            broken_edges = [(t1, t2), (t3, t4), (t5, t6)]
            delta_p = calculate_penalty_delta_fast(broken_edges, curr_tour, load_state, waste, capacity, n_original)
        else:
            delta_p = 0.0

        delta_c = best_gain

        # Step 3: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue  # Skip expensive tour construction

        # Step 4: LAZY EVALUATION
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=3, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        p3, c3 = get_score(new_tour, d, waste, capacity, n_original)

        if is_better(p3, c3, curr_p, curr_c):
            return new_tour, p3, c3, True

    return None, 0.0, 0.0, False


def _try_4opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[object] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 4-opt move with O(1) lexicographic pre-screening.

    **CRITICAL FIX**: Moves _apply_kopt_via_operator() call AFTER lexicographic
    gate to avoid O(N^5) → O(N^4) complexity reduction.

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for O(1) penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[l + 1]

        # Step 1: Compute distance gains (O(1))
        gains = _4opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, d)
        best_gain = max(gains) if gains else 0.0

        # Step 2: O(1) penalty delta via load_state
        if load_state is not None and waste is not None and capacity is not None:
            from logic.src.policies.lin_kernighan_helsgaun_three.load_tracker import (
                calculate_penalty_delta_fast,
            )

            broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)]
            delta_p = calculate_penalty_delta_fast(broken_edges, curr_tour, load_state, waste, capacity, n_original)
        else:
            delta_p = 0.0

        delta_c = best_gain

        # Step 3: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue  # Skip expensive tour construction

        # Step 4: LAZY EVALUATION
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=4, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        p4, c4 = get_score(new_tour, d, waste, capacity, n_original)

        if is_better(p4, c4, curr_p, curr_c):
            return new_tour, p4, c4, True

    return None, 0.0, 0.0, False


def _try_5opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[object] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 5-opt move with O(1) lexicographic pre-screening.

    **CRITICAL FIX**: Moves _apply_kopt_via_operator() call AFTER lexicographic
    gate to avoid O(N^6) complexity explosion.

    Algorithm:
    1. Compute distance gains (O(1))
    2. Identify broken edges and compute penalty delta via load_state (O(1))
    3. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
    4. LAZY EVALUATION: Only construct new tour if gate passes (O(N))

    Complexity: O(N^5) inner loop × O(1) gate × rare O(N) tour construction
                = O(N^5) amortized instead of O(N^6)

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for O(1) penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[m + 1]

        # Step 1: Compute distance gains (O(1))
        gains = _5opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, d)
        best_gain = max(gains) if gains else 0.0

        # Step 2: O(1) penalty delta via load_state (if available)
        if load_state is not None and waste is not None and capacity is not None:
            from logic.src.policies.lin_kernighan_helsgaun_three.load_tracker import (
                calculate_penalty_delta_fast,
            )

            # Identify broken edges for 5-opt
            broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8), (t9, t10)]

            # Fast penalty delta (O(1) relative to N)
            delta_p = calculate_penalty_delta_fast(broken_edges, curr_tour, load_state, waste, capacity, n_original)
        else:
            # Fallback: assume no penalty change (pure TSP)
            delta_p = 0.0

        delta_c = best_gain  # Positive = cost reduction

        # Step 3: Lexicographic gate
        # Accept if: (ΔP < -ε) OR (|ΔP| ≤ ε AND ΔC > ε)
        gate_passed = _should_accept_kopt_move(delta_p, delta_c)

        if not gate_passed:
            continue  # CRITICAL: Skip expensive tour construction!

        # Step 4: LAZY EVALUATION - only construct tour if gate passed
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=5, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        # Verify actual improvement (gate is conservative estimate)
        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)

        # Final check: strictly better under lexicographic objective
        if is_better(p5, c5, curr_p, curr_c):
            return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
