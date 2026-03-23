"""
LKH-3 Objective Module.

Provides the lexicographic (penalty, cost) objective used throughout the
Lin-Kernighan-Helsgaun heuristic and all supporting helper modules.

In LKH-3 (Helsgaun 2017) a solution is evaluated by a two-level criterion:
feasibility first, optimality second.  The *penalty* measures total capacity
violation — the sum of excess demand over all VRP route segments — and is
always minimised before tour cost is considered.  For pure TSP instances the
penalty is always zero and the objective reduces to plain cost minimisation.

Public API
----------
calculate_penalty(tour, waste, capacity) -> float
    Scan a tour and sum all capacity overloads.

get_score(tour, distance_matrix, waste, capacity) -> (penalty, cost)
    Return the full (penalty, cost) pair for a closed or open tour.

is_better(p1, c1, p2, c2) -> bool
    Lexicographic dominance check: True iff (p1, c1) strictly beats (p2, c2).

Typical usage
-------------
>>> from logic.src.policies.other.operators.heuristics._objective import (
...     get_score, is_better
... )
>>> pen, cost = get_score(tour, dist, demands, capacity)
>>> if is_better(pen, cost, best_pen, best_cost):
...     best_tour = tour[:]
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Penalty / objective helpers (LKH-3 lexicographic objective)
# ---------------------------------------------------------------------------


def calculate_penalty(
    tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> float:
    """
    Compute total capacity-violation penalty for a VRP tour.

    The tour is encoded as a sequence that visits node 0 (depot) one or more
    times.  Each time node 0 is encountered the vehicle load resets to 0.

    Args:
        tour: Node sequence (open or closed; depot = 0).
        waste: 1-D array of node demands (index 0 = depot demand, usually 0).
        capacity: Vehicle capacity.

    Returns:
        Total excess demand summed over all route segments.  Zero for TSP.
    """
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
    Evaluate a tour's (penalty, cost) under the LKH-3 lexicographic objective.

    Args:
        tour: Closed or open node sequence.
        distance_matrix: (n × n) cost matrix.
        waste: Node demands (or None for pure TSP).
        capacity: Vehicle capacity (or None for pure TSP).

    Returns:
        (penalty, cost) tuple.
    """
    n = len(tour) - 1
    c = 0.0
    for i in range(n):
        c += distance_matrix[tour[i], tour[i + 1]]
    pen = calculate_penalty(tour, waste, capacity)
    return pen, c


def is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """
    Lexicographic comparison: penalty first, then cost.

    Returns True iff (p1, c1) strictly dominates (p2, c2).
    """
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def is_better_or_equal(p1: float, c1: float, p2: float, c2: float) -> bool:
    """
    Non-strict lexicographic dominance: penalty first, then cost.

    Returns True iff (p1, c1) is at least as good as (p2, c2) under the
    lexicographic ordering.  Used by LKH-3 for infeasible-transit moves
    where the search is allowed to accept moves that *maintain* the current
    objective while transiting through infeasible space.

    The key difference from :func:`is_better`:

    - ``is_better``:          strictly better (used for best-solution updates)
    - ``is_better_or_equal``: weakly better   (used for move acceptance in
      the local-search loop to allow lateral moves through infeasible space)
    """
    if p1 < p2 - 1e-6:
        return True
    if abs(p1 - p2) > 1e-6:
        return False
    return c1 <= c2 + 1e-6


def penalty_delta(
    old_tour: List[int],
    new_tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> float:
    """
    Compute the change in penalty: ΔP = P(new) − P(old).

    A negative value means the new tour is *less* infeasible.  This is
    used by the LKH-3 lexicographic objective to allow moves that reduce
    constraint violation even if they increase routing cost.

    For pure TSP instances (waste/capacity is None) the delta is always 0.

    Args:
        old_tour: Current tour (closed or open).
        new_tour: Proposed tour (closed or open).
        waste: 1-D demand array.
        capacity: Vehicle capacity.

    Returns:
        P(new_tour) − P(old_tour).
    """
    if waste is None or capacity is None:
        return 0.0
    return calculate_penalty(new_tour, waste, capacity) - calculate_penalty(old_tour, waste, capacity)
