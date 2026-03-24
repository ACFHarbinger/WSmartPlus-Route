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

from logic.src.policies.lin_kernighan_helsgaun_three.kopt_topologies import (
    EXHAUSTIVE_2OPT_CASES,
    EXHAUSTIVE_3OPT_CASES,
    EXHAUSTIVE_4OPT_CASES,
    EXHAUSTIVE_5OPT_CASES,
)
from logic.src.policies.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
    get_exact_penalty_delta,
)
from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
    get_score,
    is_better,
)
from logic.src.policies.lin_kernighan_helsgaun_three.tour_construction import (
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

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_2OPT_CASES constant
    (1 valid topology) for architectural consistency with higher-order k-opt.

    Algorithm (Lazy Evaluation Pattern):
    1. For each candidate t3 of t2:
       a. Extract broken nodes [t1, t2, t3, t4]
       b. Compute distance gain ΔC in O(1) using the single cached topology
       c. Compute penalty delta ΔP via load_state — O(L)
       d. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
       e. ONLY IF gate passes: construct tour and verify

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.

    Complexity:
        - Per candidate: O(1) distance lookup + O(L) penalty check
        - Tour construction: O(N) only when gate passes
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Get current tour score for final verification
    p_curr, c_curr = get_score(curr_tour, d, waste, capacity, n_original)

    # Extract the single cached 2-opt topology (for architectural consistency)
    # EXHAUSTIVE_2OPT_CASES = [[(0, 2), (1, 3)]]
    assert len(EXHAUSTIVE_2OPT_CASES) == 1, "2-opt must have exactly 1 topology"
    topology_2opt = EXHAUSTIVE_2OPT_CASES[0]

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

        # The 4 endpoints of the 2 broken edges (indexed 0..3 in cached topology)
        broken_nodes = [t1, t2, t3, t4]
        broken_edges = [(t1, t2), (t3, t4)]

        # Phase 1: Compute distance gain using cached topology
        # topology_2opt = [(0, 2), (1, 3)] maps to [(t1, t3), (t2, t4)]
        added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology_2opt]
        base_cost = d[t1, t2] + d[t3, t4]
        added_cost = sum(d[u, v] for u, v in added_edges)
        delta_c = base_cost - added_cost

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                added_edges,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0
            if delta_c <= 1e-6:
                continue

        # Phase 3: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        new_tour = build_tour_from_segments(
            curr_tour,
            [i, i + 1, j, j + 1 if j + 1 < nodes_count else 0],
            topology_2opt,
            k=2,
        )

        # Phase 5: Full verification
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
    Search for an improving 3-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_3OPT_CASES constant
    (7 valid topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each third cut position k_pos:
       a. Extract the 6 broken nodes [t1, t2, t3, t4, t5, t6]
       b. Iterate over ALL 7 cached 3-opt topologies
       c. Map cached indices [0..5] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 7 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Note: The 7 cached cases include 2 pure 2-opt sub-moves. These are
    evaluated anyway since the constant-factor overhead is negligible and
    they may find improving moves that were missed due to α-nearest neighbor
    filtering in the preceding _try_2opt_move search.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improvement found.

    Complexity:
        - Per third-cut position: O(7) = O(1) topology evaluations
        - Distance lookups: O(7 × 3) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid third cut positions
    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[k_pos + 1]

        # The 6 endpoints of the 3 broken edges (indexed 0..5 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 7 cached 3-opt topologies
        for topology in EXHAUSTIVE_3OPT_CASES:
            # Map cached indices [0..5] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..5]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 3 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 7 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        pos = [i, i + 1, j, j + 1, k_pos, k_pos + 1 if k_pos + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=3)

        # Phase 5: Full verification
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
    Search for an improving 4-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_4OPT_CASES constant
    (25 valid non-sequential topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each fourth cut position l:
       a. Extract the 8 broken nodes [t1, t2, t3, t4, t5, t6, t7, t8]
       b. Iterate over ALL 25 cached 4-opt topologies
       c. Map cached indices [0..7] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 25 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improvement found.

    Complexity:
        - Per fourth-cut position: O(25) = O(1) topology evaluations
        - Distance lookups: O(25 × 4) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid fourth cut positions
    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[l + 1]

        # The 8 endpoints of the 4 broken edges (indexed 0..7 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 25 cached 4-opt topologies
        for topology in EXHAUSTIVE_4OPT_CASES:
            # Map cached indices [0..7] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..7]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 4 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 25 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,  # type: ignore[arg-type]
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        pos = [i, i + 1, j, j + 1, k, k + 1, l, l + 1 if l + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=4)

        # Phase 5: Full verification
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
    Search for an improving 5-opt move with exhaustive topology enumeration.

    **CRITICAL UPGRADE**: This function now evaluates ALL 208 valid non-sequential
    5-opt reconnection topologies using the module-level cached constant
    EXHAUSTIVE_5OPT_CASES. The previous implementation evaluated only 5
    representative cases, which was mathematically incomplete for publication.

    Algorithm:
    1. For each fifth cut position m:
       a. Extract the 10 broken nodes [t1, t2, ..., t10]
       b. Iterate over ALL 208 cached 5-opt topologies
       c. Map cached indices [0..9] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 208 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Complexity:
        - Per fifth-cut position: O(208) = O(1) topology evaluations
        - Distance lookups: O(208 × 5) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
        - Overall: O(N^5) amortized (same asymptotic as before, more thorough)

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.

    References:
        Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
        traveling salesman heuristic. European Journal of Operational Research,
        126(1), 106-130.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid fifth cut positions
    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[m + 1]

        # The 10 endpoints of the 5 broken edges (indexed 0..9 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8), (t9, t10)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 208 cached 5-opt topologies
        for topology in EXHAUSTIVE_5OPT_CASES:
            # Map cached indices [0..9] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..9]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 5 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 208 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        pos = [i, i + 1, j, j + 1, k, k + 1, l, l + 1, m, m + 1 if m + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos, best_topology, k=5)

        # Phase 5: Full verification
        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p5, c5, curr_p, curr_c):
            return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
