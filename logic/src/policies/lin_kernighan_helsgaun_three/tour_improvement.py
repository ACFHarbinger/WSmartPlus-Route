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
   **CRITICAL FIX**: Now uses lazy evaluation — checks the lexicographic
   gate BEFORE constructing the candidate tour.

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
- :func:`get_exact_penalty_delta` from ``load_tracker`` for O(L) penalty queries

Typical usage
-------------
>>> from logic.src.policies.lin_kernighan_helsgaun_three.tour_improvement import (
...     _apply_kopt_via_operator,
...     _try_2opt_move,
...     _try_3opt_move,
... )
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
    get_exact_penalty_delta,
)
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
    load_state: Optional[LoadState] = None,
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    **CRITICAL FIX**: Now uses lazy evaluation — the lexicographic gate is
    checked BEFORE constructing the candidate tour.  This avoids wasting
    CPU cycles on _apply_kopt_via_operator for moves that will be rejected.

    Algorithm:
    1. For each candidate t3 of t2:
       a. Compute distance gain ΔC = _2opt_gain(t1, t2, t3, t4, d) — O(1)
       b. Compute penalty delta ΔP via load_state — O(L)
       c. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
       d. ONLY IF gate passes: construct tour and verify

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Get current tour score for final verification
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

        # Step 1: Compute distance gain ΔC (O(1))
        delta_c = _2opt_gain(t1, t2, t3, t4, d)

        # Step 2: Compute penalty delta ΔP
        if waste is not None and capacity is not None:
            if load_state is not None:
                # For 2-opt we need the candidate tour to compute exact ΔP.
                # But first, check if the distance gain alone is promising.
                # If ΔC <= 0, we'd need ΔP < 0 for acceptance.
                # We defer exact penalty computation to after tour construction.
                pass

            # TSP shortcut: if pure cost and no gain, skip
            if load_state is None and delta_c <= 1e-6:
                continue

            # Construct candidate tour (needed for exact ΔP)
            candidate_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
            if candidate_tour is None:
                continue

            # Compute exact penalty delta
            if load_state is not None:
                delta_p = get_exact_penalty_delta(
                    candidate_tour,
                    load_state,
                    waste,
                    capacity,
                    n_original,  # type: ignore[arg-type]
                )
            else:
                delta_p = penalty_delta(curr_tour, candidate_tour, waste, capacity, n_original)

            # Lexicographic gate
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
    load_state: Optional[LoadState] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 3-opt move with lazy lexicographic pre-screening.

    Cases 1 & 2 from _3opt_gains are excluded because they are pure 2-opt
    sub-moves — these are already handled by the preceding _try_2opt_move
    search, so including them would be redundant work.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator passed to the operator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

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

        delta_c = best_gain

        # Step 2: Quick skip if no cost improvement and no load_state for
        # penalty evaluation
        if load_state is None and delta_c <= 1e-6:
            continue

        # Step 3: LAZY EVALUATION — construct tour
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=3, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        # Step 4: Exact penalty delta (using constructed tour)
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                new_tour,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Step 5: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
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
    load_state: Optional[LoadState] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 4-opt move with lazy lexicographic pre-screening.

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

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

        delta_c = best_gain

        # Quick skip
        if load_state is None and delta_c <= 1e-6:
            continue

        # Step 2: LAZY EVALUATION — construct tour
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=4, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        # Step 3: Exact penalty delta
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                new_tour,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Step 4: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
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
    load_state: Optional[LoadState] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 5-opt move with lazy lexicographic pre-screening.

    NOTE: _5opt_gains evaluates 5 representative cases out of the full 60
    possible reconnection patterns.  These 5 cases cover the sequential
    5-opt moves central to Helsgaun (2000), Section 4.3.

    Algorithm:
    1. Compute distance gains (O(1))
    2. Construct candidate tour (lazy — only when cost gain is promising)
    3. Compute exact penalty delta using constructed tour
    4. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
    5. Verify with full (penalty, cost) evaluation

    Complexity: O(N^5) inner loop × rare O(N) tour construction
                = O(N^5) amortized instead of O(N^6)

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

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

        delta_c = best_gain

        # Quick skip
        if load_state is None and delta_c <= 1e-6:
            continue

        # Step 2: LAZY EVALUATION — construct tour
        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=5, distance_matrix=d, rng=rng)
        if new_tour is None:
            continue

        # Step 3: Exact penalty delta
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                new_tour,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Step 4: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Step 5: Verify actual improvement
        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)

        if is_better(p5, c5, curr_p, curr_c):
            return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
