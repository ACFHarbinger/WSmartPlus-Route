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
)
from logic.src.policies.lin_kernighan_helsgaun_three.tour_construction import (
    KoptTopologyFactory,
    _should_accept_kopt_move,
    build_tour_from_segments,
)

# ---------------------------------------------------------------------------
# Inner search routines with gain pre-screening
# ---------------------------------------------------------------------------


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

        # Step 1: Distance gain ΔC
        delta_c = d[t1, t2] + d[t3, t4] - d[t1, t3] - d[t2, t4]

        # Step 2: Compute penalty delta ΔP virtually
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                [(t1, t2), (t3, t4)],
                [(t1, t3), (t2, t4)],
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0
            if delta_c <= 1e-6:
                continue

        # Step 3: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Step 4: Construct tour EXACTLY using topology
        new_tour = build_tour_from_segments(
            curr_tour,
            [i, i + 1, j, j + 1 if j + 1 < nodes_count else 0],
            [(0, 2), (1, 3)],  # Exact 2-opt topology matching
            k=2,
        )

        p2, c2 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p2, c2, p_curr, c_curr):
            return new_tour, p2, c2, True, j

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

        broken_nodes = [t1, t2, t3, t4, t5, t6]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6]

        topologies = KoptTopologyFactory.get_topologies(3)

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in topologies:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # Quick skip
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Exact penalty delta virtually evaluated
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Lexicographic gate
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Construct and verify
        pos = [i, i + 1, j, j + 1, k_pos, k_pos + 1 if k_pos + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=3)

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

        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]

        topologies = KoptTopologyFactory.get_topologies(4)

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in topologies:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # Quick skip
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Exact penalty delta virtually evaluated
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,
                load_state,  # type: ignore[arg-type]
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Lexicographic gate
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Construct and verify
        pos = [i, i + 1, j, j + 1, k, k + 1, l, l + 1 if l + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=4)

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

        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8), (t9, t10)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]

        topologies = KoptTopologyFactory.get_topologies(5)

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in topologies:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # Quick skip
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Exact penalty delta virtually evaluated
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Lexicographic gate
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Construct and verify
        pos = [i, i + 1, j, j + 1, k, k + 1, l, l + 1, m, m + 1 if m + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=5)

        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p5, c5, curr_p, curr_c):
            return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
