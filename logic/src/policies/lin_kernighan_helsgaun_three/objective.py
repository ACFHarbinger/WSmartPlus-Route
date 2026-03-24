"""
LKH-3 Objective Module.

Provides the lexicographic (penalty, cost) objective used throughout the
Lin-Kernighan-Helsgaun heuristic and all supporting helper modules.

In LKH-3 (Helsgaun 2017) a solution is evaluated by a two-level criterion:
feasibility first, optimality second.  The *penalty* measures total capacity
violation — the sum of excess demand over all VRP route segments — and is
always minimised before tour cost is considered.  For pure TSP instances the
penalty is always zero and the objective reduces to plain cost minimisation.

**Augmented Dummy Depot Encoding (Current Standard)**:
To enable native multi-route optimization, the graph is augmented with
explicit dummy depot nodes at indices [N, N+1, ..., N+M-2] where N is the
original graph size and M is the number of vehicles.

Example (N=5 original nodes, M=3 vehicles):
  - Original nodes: [0, 1, 2, 3, 4]  (0 = depot, 1-4 = customers)
  - Augmented graph: [0, 1, 2, 3, 4, 5, 6]  (5, 6 = dummy depots)
  - Tour: [0, 3, 1, 5, 2, 4, 6, 0]  (split at indices >= 5)
  - Routes: [[3, 1], [2, 4]]

This avoids NumPy negative-indexing bugs and provides O(1) array access.

**Legacy Mode (Deprecated)**:
Old code used negative indices (-1, -2, -3) for dummy depots. This is
maintained for backward compatibility via n_original=None parameter, but
causes catastrophic NumPy indexing failures. Use augmented mode instead!

Public API
----------
calculate_penalty(tour, waste, capacity) -> float
    Scan a tour and sum all capacity overloads (supports dummy depots).

get_score(tour, distance_matrix, waste, capacity) -> (penalty, cost)
    Return the full (penalty, cost) pair for a tour with dummy depots.

is_better(p1, c1, p2, c2) -> bool
    Lexicographic dominance check: True iff (p1, c1) strictly beats (p2, c2).

split_tour_at_dummies(tour) -> List[List[int]]
    Extract multi-route representation from a dummy-depot-encoded tour.

inject_dummy_depots(routes, n_vehicles) -> List[int]
    Inject dummy depots into a flat tour to enable multi-route k-opt.

Typical usage
-------------
>>> from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
...     get_score, is_better, split_tour_at_dummies
... )
>>> pen, cost = get_score(tour, dist, demands, capacity)
>>> if is_better(pen, cost, best_pen, best_cost):
...     routes = split_tour_at_dummies(tour)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Dummy Depot Constants
# ---------------------------------------------------------------------------

# DEPRECATED: Negative-index dummy depots (causes NumPy indexing bugs)
DUMMY_DEPOT_START = -1  # Legacy only - use augmented mode instead!

# Current standard: Augmented dummy depots at indices [N, N+1, ..., N+M-2]
DEPOT_NODE = 0  # Main depot (always index 0)

# ---------------------------------------------------------------------------
# Dummy Depot Utility Functions
# ---------------------------------------------------------------------------


def is_dummy_depot(node: int, n_original: Optional[int] = None) -> bool:
    """
    Check if a node is a dummy depot.

    Supports both legacy negative-index encoding and new augmented encoding.

    Args:
        node: Node index to check.
        n_original: Original graph size (for augmented mode). If None,
                    uses legacy negative-index check.

    Returns:
        True if node is a dummy depot, False otherwise.

    Examples:
        >>> # Legacy mode (negative indices)
        >>> is_dummy_depot(-1)
        True
        >>> is_dummy_depot(5)
        False
        >>>
        >>> # Augmented mode (indices >= n_original)
        >>> is_dummy_depot(5, n_original=5)
        True
        >>> is_dummy_depot(3, n_original=5)
        False
    """
    if n_original is not None:
        # Augmented mode: dummy depots are indices >= n_original
        return node >= n_original
    else:
        # Legacy mode: dummy depots are negative indices
        return node < 0


def split_tour_at_dummies(tour: List[int], n_original: Optional[int] = None) -> List[List[int]]:
    """
    Extract multi-route representation from a dummy-depot-encoded tour.

    Splits the tour at every occurrence of a real depot (0) or dummy depot.
    Each sub-route contains only customer nodes.

    Args:
        tour: Closed tour with dummy depot markers.
              Legacy example: [0, 3, 5, -1, 7, 2, -2, 9, 0]
              Augmented example: [0, 3, 5, 6, 7, 2, 7, 9, 0] (n_original=6)
        n_original: Original graph size (for augmented mode). If None,
                    uses legacy negative-index detection.

    Returns:
        List of routes (no depot nodes).
        Example: [[3, 5], [7, 2], [9]]
    """
    routes: List[List[int]] = []
    current: List[int] = []

    for node in tour:
        if node == DEPOT_NODE or is_dummy_depot(node, n_original):
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)

    if current:
        routes.append(current)

    return routes


def inject_dummy_depots(routes: List[List[int]]) -> List[int]:
    """
    Inject dummy depots into a multi-route solution to create a flat tour.

    Dummy depots are inserted between consecutive routes to mark vehicle boundaries.
    The resulting tour can be processed by k-opt operators that treat dummies as
    swappable nodes, enabling dynamic route rebalancing.

    Args:
        routes: List of routes (no depot nodes).

    Returns:
        Closed tour with dummy depots.
        Example: routes=[[3,5], [7,2], [9]] → [0, 3, 5, -1, 7, 2, -2, 9, 0]
    """
    if not routes:
        return [DEPOT_NODE, DEPOT_NODE]

    tour = [DEPOT_NODE]

    for idx, route in enumerate(routes):
        tour.extend(route)
        if idx < len(routes) - 1:
            # Insert dummy depot: -1, -2, -3, ...
            dummy_id = DUMMY_DEPOT_START - idx
            tour.append(dummy_id)

    tour.append(DEPOT_NODE)
    return tour


# ---------------------------------------------------------------------------
# Penalty / objective helpers (LKH-3 lexicographic objective)
# ---------------------------------------------------------------------------


def calculate_penalty(
    tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int] = None,
) -> float:
    """
    Compute total capacity-violation penalty for a VRP tour with dummy depots.

    The tour is scanned left-to-right. Each time a depot (real or dummy) is
    encountered, the vehicle load resets to 0. Dummy depots have zero demand.

    Args:
        tour: Node sequence with possible dummy depot markers.
              Legacy: [0, 3, 5, -1, 7, 2, 0]
              Augmented: [0, 3, 5, 6, 7, 2, 0] (n_original=6)
        waste: 1-D array of node demands (index 0 = depot demand, usually 0).
               For augmented mode, includes zero demands for dummy depots.
        capacity: Vehicle capacity.
        n_original: Original graph size (for augmented mode). If None,
                    uses legacy negative-index detection.

    Returns:
        Total excess demand summed over all route segments.  Zero for TSP.
    """
    if waste is None or capacity is None:
        return 0.0

    penalty = 0.0
    current_load = 0.0

    for node in tour:
        if node == DEPOT_NODE or is_dummy_depot(node, n_original):
            # Reset load at any depot (real or dummy)
            current_load = 0.0
        else:
            # Accumulate demand for customer nodes
            if 0 <= node < len(waste):
                current_load += waste[node]
                if current_load > capacity + 1e-6:
                    penalty += current_load - capacity

    return penalty


def get_score(
    tour: List[int],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate a tour's (penalty, cost) under the LKH-3 lexicographic objective.

    Handles tours with dummy depots. In augmented mode, dummy depot distances
    are directly available in the distance matrix. In legacy mode, they map to
    the main depot.

    Args:
        tour: Closed or open node sequence, possibly with dummy depot markers.
        distance_matrix: (n × n) cost matrix.
        waste: Node demands (or None for pure TSP).
        capacity: Vehicle capacity (or None for pure TSP).
        n_original: Original graph size (for augmented mode). If None,
                    uses legacy mapping to depot.

    Returns:
        (penalty, cost) tuple.
    """
    n = len(tour) - 1
    c = 0.0
    for i in range(n):
        curr_node = tour[i]
        next_node = tour[i + 1]

        # Legacy mode: Map dummy depots to main depot for distance calculation
        if n_original is None:
            if is_dummy_depot(curr_node, None):
                curr_node = DEPOT_NODE
            if is_dummy_depot(next_node, None):
                next_node = DEPOT_NODE

        # Add edge cost (augmented mode uses direct indices)
        if 0 <= curr_node < len(distance_matrix) and 0 <= next_node < len(distance_matrix):
            c += distance_matrix[curr_node, next_node]

    pen = calculate_penalty(tour, waste, capacity, n_original)
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
    n_original: Optional[int] = None,
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
        n_original: Original graph size (for augmented mode).

    Returns:
        P(new_tour) − P(old_tour).
    """
    if waste is None or capacity is None:
        return 0.0
    return calculate_penalty(new_tour, waste, capacity, n_original) - calculate_penalty(
        old_tour, waste, capacity, n_original
    )
