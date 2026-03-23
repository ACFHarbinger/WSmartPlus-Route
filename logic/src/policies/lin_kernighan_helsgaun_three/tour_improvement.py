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
    is_better_or_equal,
)
from logic.src.policies.lin_kernighan_helsgaun_three.tour_adapter import TourAdapter
from logic.src.policies.lin_kernighan_helsgaun_three.tour_construction import (
    _2opt_gain,
    _3opt_gains,
    _4opt_gains,
    _5opt_gains,
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
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    Iterates over α-nearest neighbours of t2 as the candidate for t3.
    Uses an exact gain pre-screen (:func:`_2opt_gain`) to skip clearly
    non-improving pairs, then delegates the actual move to
    :func:`move_kopt_intra` (k=2) via :func:`_apply_kopt_via_operator`.

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator.

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

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

        # Exact gain pre-screen: only call the operator when gain > 0
        gain = _2opt_gain(t1, t2, t3, t4, d)
        if gain > 1e-6:
            new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
            if new_tour is not None:
                p_new, c_new = get_score(new_tour, d, waste, capacity)
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
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 3-opt move extending the (t1,t2), (t3,t4) pair.

    For each valid third cut position k_pos (t5, t6), computes exact gains
    (:func:`_3opt_gains`) for all seven non-trivial reconnection cases.
    When at least one non-redundant case has a positive gain the move is
    delegated to :func:`move_kopt_intra` (k=3) via
    :func:`_apply_kopt_via_operator`, which selects the best of the patterns
    it evaluates at a sampled third cut.

    Cases 1 and 2 (pure 2-opt sub-moves) are excluded from the pre-screen to
    avoid redundant work with the 2-opt phase.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator passed to the operator.

    Returns:
        (new_tour, penalty, cost, improved).
        Returns (None, 0, 0, False) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity)

    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[k_pos + 1]

        gains = _3opt_gains(t1, t2, t3, t4, t5, t6, d)
        # Cases 1 & 2 are pure 2-opt on one sub-segment; skip to avoid duplication
        improving = any(g > 1e-6 for idx, g in enumerate(gains) if idx not in (1, 2))
        if not improving:
            continue

        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=3, distance_matrix=d, rng=rng)
        if new_tour is not None:
            p3, c3 = get_score(new_tour, d, waste, capacity)
            # LKH-3: accept if strictly better OR penalty decreases
            # (infeasible transit — allows escaping local minima)
            if is_better(p3, c3, curr_p, curr_c) or (p3 < curr_p - 1e-6 and is_better_or_equal(p3, c3, curr_p, curr_c)):
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
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 4-opt move extending three broken edges.

    For each valid fourth cut position l (t7, t8), computes exact gains
    (:func:`_4opt_gains`) for three common 4-opt patterns and delegates to
    :func:`move_kopt_intra` (k=4) via :func:`_apply_kopt_via_operator` when
    at least one pattern is improving.

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.

    Returns:
        (new_tour, penalty, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity)

    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[l + 1]

        gains = _4opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, d)
        if not any(g > 1e-6 for g in gains):
            continue

        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=4, distance_matrix=d, rng=rng)
        if new_tour is not None:
            p4, c4 = get_score(new_tour, d, waste, capacity)
            # LKH-3: accept if strictly better OR penalty decreases
            if is_better(p4, c4, curr_p, curr_c) or (p4 < curr_p - 1e-6 and is_better_or_equal(p4, c4, curr_p, curr_c)):
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
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 5-opt move extending four broken edges.

    Implements Helsgaun (2000) Section 4.3: the basic LKH move is a
    sequential 5-opt.  For each valid fifth cut position m (t9, t10), exact
    gains (:func:`_5opt_gains`) are computed; when at least one is positive
    the move is delegated to :func:`move_kopt_intra` (k=5) via
    :func:`_apply_kopt_via_operator`.

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator.

    Returns:
        (new_tour, penalty, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity)

    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[m + 1]

        gains = _5opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, d)
        if not any(g > 1e-6 for g in gains):
            continue

        new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=5, distance_matrix=d, rng=rng)
        if new_tour is not None:
            p5, c5 = get_score(new_tour, d, waste, capacity)
            # LKH-3: accept if strictly better OR penalty decreases
            if is_better(p5, c5, curr_p, curr_c) or (p5 < curr_p - 1e-6 and is_better_or_equal(p5, c5, curr_p, curr_c)):
                return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
