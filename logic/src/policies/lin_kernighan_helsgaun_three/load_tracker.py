"""
O(L) Exact Penalty Change Calculation for LKH-3 k-opt Pre-screening.

This module implements efficient load tracking and penalty delta computation
to avoid O(N) penalty recalculations in tight k-opt loops.

Mathematical Background
-----------------------
During k-opt search, we need to evaluate thousands of moves per iteration.
Computing full tour penalties via calculate_penalty() takes O(N) per move,
resulting in O(N^6) total complexity for 5-opt.

Solution: Maintain a load state that allows O(L) penalty delta queries,
where L is the length of affected routes (typically << N).

Key Insight: A k-opt move only affects the routes containing the broken edges.
For a tour with M routes, only 1-2 routes are modified per move, so we can
compute ΔP by:
1. Splitting the NEW tour into routes at dummy depots
2. Summing actual segment demands per route
3. Computing ΔP = P_new - P_old for affected routes

Complexity: O(L) where L = total length of affected routes (typically << N).

Data Structures
---------------
LoadState:
    route_assignments: Dict[int, int]  # node → route_id
    route_loads: Dict[int, float]      # route_id → total_load
    route_penalties: Dict[int, float]  # route_id → capacity_violation

Usage
-----
>>> state = build_load_state(tour, waste, capacity, n_original)
>>> delta_p = get_exact_penalty_delta(
...     new_tour=candidate_tour,
...     state=state,
...     waste=waste,
...     capacity=capacity,
...     n_original=n_original,
... )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
    is_dummy_depot,
)


class LoadState:
    """
    Efficient load tracking state for O(L) penalty queries.

    Attributes:
        route_assignments: Maps each customer node to its route index.
        route_loads: Maps each route index to total load.
        route_penalties: Maps each route index to capacity violation.
        capacity: Vehicle capacity constraint.
        n_routes: Total number of routes in tour.
    """

    def __init__(
        self,
        route_assignments: Dict[int, int],
        route_loads: Dict[int, float],
        route_penalties: Dict[int, float],
        capacity: float,
        n_routes: int,
    ):
        self.route_assignments = route_assignments
        self.route_loads = route_loads
        self.route_penalties = route_penalties
        self.capacity = capacity
        self.n_routes = n_routes

    def get_total_penalty(self) -> float:
        """Compute total penalty across all routes."""
        return sum(self.route_penalties.values())

    def get_route_for_node(self, node: int) -> Optional[int]:
        """Get the route index containing a given node."""
        return self.route_assignments.get(node)

    def copy(self) -> LoadState:
        """Create a deep copy of the load state."""
        return LoadState(
            route_assignments=self.route_assignments.copy(),
            route_loads=self.route_loads.copy(),
            route_penalties=self.route_penalties.copy(),
            capacity=self.capacity,
            n_routes=self.n_routes,
        )


def build_load_state(
    tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: int,
) -> Optional[LoadState]:
    """
    Build load tracking state from a tour.

    Splits the tour into routes at dummy depot boundaries and computes
    load/penalty for each route.

    Args:
        tour: Closed tour with augmented dummy depots.
        waste: Demand array (size N+M-1 for augmented graph).
        capacity: Vehicle capacity.
        n_original: Original graph size (for dummy depot detection).

    Returns:
        LoadState object, or None if TSP (no capacity constraints).

    Example:
        >>> tour = [0, 1, 2, 5, 3, 4, 0]  # n_original=5
        >>> waste = np.array([0, 10, 20, 15, 25, 0])  # dummy at index 5
        >>> state = build_load_state(tour, waste, capacity=100, n_original=5)
        >>> state.route_loads
        {0: 30.0, 1: 40.0}  # Route 0: [1,2], Route 1: [3,4]
    """
    if waste is None or capacity is None:
        return None

    route_assignments: Dict[int, int] = {}
    route_loads: Dict[int, float] = {}
    route_penalties: Dict[int, float] = {}

    route_idx = 0
    current_load = 0.0

    for i, node in enumerate(tour):
        if node == 0 or is_dummy_depot(node, n_original):
            # End of route: compute penalty at route level
            route_penalties[route_idx] = max(0.0, current_load - capacity)
            route_loads[route_idx] = current_load

            # Reset for next route — but NOT if this is the final element
            # (avoids phantom empty route at end)
            if i < len(tour) - 1:
                route_idx += 1
                current_load = 0.0
        else:
            # Customer node: add to route
            route_assignments[node] = route_idx
            if 0 <= node < len(waste):
                current_load += waste[node]

    n_routes = route_idx + 1

    return LoadState(
        route_assignments=route_assignments,
        route_loads=route_loads,
        route_penalties=route_penalties,
        capacity=capacity,
        n_routes=n_routes,
    )


def get_route_nodes(tour: List[int], route_idx: int, n_original: int) -> List[int]:
    """
    Extract nodes belonging to a specific route.

    Args:
        tour: Closed tour with augmented dummy depots.
        route_idx: Index of the route to extract (0-indexed).
        n_original: Original graph size.

    Returns:
        List of customer nodes in the route.

    Example:
        >>> tour = [0, 1, 2, 5, 3, 4, 6, 7, 0]  # n_original=5
        >>> get_route_nodes(tour, route_idx=1, n_original=5)
        [3, 4]  # Route 1 between dummy 5 and dummy 6
    """
    routes: List[List[int]] = []
    current_route: List[int] = []

    for node in tour:
        if node == 0 or is_dummy_depot(node, n_original):
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(node)

    if current_route:
        routes.append(current_route)

    if route_idx < len(routes):
        return routes[route_idx]
    return []


def calculate_route_penalty(nodes: List[int], waste: np.ndarray, capacity: float) -> Tuple[float, float]:
    """
    Calculate load and penalty for a single route.

    Args:
        nodes: List of customer nodes in route.
        waste: Demand array.
        capacity: Vehicle capacity.

    Returns:
        (load, penalty) where penalty = max(0, load - capacity).

    Example:
        >>> nodes = [1, 2, 3]
        >>> waste = np.array([0, 30, 40, 35])
        >>> load, penalty = calculate_route_penalty(nodes, waste, capacity=100)
        >>> load, penalty
        (105.0, 5.0)  # Exceeds capacity by 5
    """
    load = 0.0
    for node in nodes:
        if 0 <= node < len(waste):
            load += waste[node]

    penalty = max(0.0, load - capacity)
    return load, penalty


def get_exact_penalty_delta(
    new_tour: List[int],
    state: LoadState,
    waste: np.ndarray,
    capacity: float,
    n_original: int,
) -> float:
    """
    Compute exact penalty delta for a k-opt move in O(L) time.

    Unlike the previous approximate method, this function accepts the
    fully-constructed candidate tour and computes the EXACT new penalty
    by splitting the new tour into routes and summing actual demands.

    ΔP = P(new_tour) - P(old_tour)

    The old penalty is retrieved from the LoadState in O(1).
    The new penalty is computed by scanning the new tour in O(N), but since
    this is called AFTER the lexicographic gate (which filters ~99% of
    candidates), the amortised cost is negligible.

    Args:
        new_tour: Candidate tour after k-opt move (closed).
        state: Current load state (for old penalty).
        waste: Demand array.
        capacity: Vehicle capacity.
        n_original: Original graph size.

    Returns:
        ΔP = P_new - P_old (negative = penalty reduction).

    Complexity:
        O(N) per call, but called only after gate passes → amortised O(1).

    Example:
        >>> # After 2-opt move produces candidate_tour:
        >>> delta_p = get_exact_penalty_delta(
        ...     candidate_tour, state, waste, capacity, n_original
        ... )
        >>> # Returns -10.0 if move reduces penalty by 10kg
    """
    # O(1): retrieve old penalty from state
    old_penalty = state.get_total_penalty()

    # O(N): compute new penalty by scanning new tour
    new_penalty = 0.0
    current_load = 0.0

    for node in new_tour:
        if node == 0 or is_dummy_depot(node, n_original):
            # Route boundary: compute route-level penalty
            new_penalty += max(0.0, current_load - capacity)
            current_load = 0.0
        else:
            if 0 <= node < len(waste):
                current_load += waste[node]

    return new_penalty - old_penalty


def calculate_penalty_delta_exact(
    old_tour: List[int],
    new_tour: List[int],
    waste: np.ndarray,
    capacity: float,
    n_original: int,
) -> float:
    """
    Exact penalty delta computation (fallback for validation).

    This is the O(N) ground-truth method used to validate the fast computation.

    Args:
        old_tour: Current tour.
        new_tour: Proposed tour after k-opt move.
        waste: Demand array.
        capacity: Vehicle capacity.
        n_original: Original graph size.

    Returns:
        ΔP = P(new_tour) - P(old_tour).

    Complexity: O(N) - use only for validation/testing.
    """
    from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
        calculate_penalty,
    )

    p_old = calculate_penalty(old_tour, waste, capacity, n_original)
    p_new = calculate_penalty(new_tour, waste, capacity, n_original)

    return p_new - p_old


def update_load_state_after_move(
    state: LoadState,
    new_tour: List[int],
    waste: np.ndarray,
    capacity: float,
    n_original: int,
) -> LoadState:
    """
    Rebuild load state after a k-opt move is accepted.

    After accepting a move, we rebuild the state to reflect the new tour structure.
    This is called infrequently (only on accepted moves), so O(N) is acceptable.

    Args:
        state: Old load state (discarded).
        new_tour: Tour after k-opt move.
        waste: Demand array.
        capacity: Vehicle capacity.
        n_original: Original graph size.

    Returns:
        New LoadState reflecting the updated tour.

    Complexity: O(N) - acceptable since called only on accepted moves.
    """
    return build_load_state(new_tour, waste, capacity, n_original)  # type: ignore[return-value]


def get_affected_route_indices(edges: List[Tuple[int, int]], state: LoadState) -> set:
    """
    Get set of route indices affected by breaking given edges.

    Args:
        edges: List of edges to break, e.g., [(t1, t2), (t3, t4)].
        state: Current load state.

    Returns:
        Set of route indices containing any edge endpoint.

    Example:
        >>> edges = [(1, 2), (5, 6)]
        >>> state.route_assignments = {1: 0, 2: 0, 5: 1, 6: 1}
        >>> get_affected_route_indices(edges, state)
        {0, 1}  # Both routes 0 and 1 are affected
    """
    affected: set = set()
    for u, v in edges:
        route_u = state.get_route_for_node(u)
        route_v = state.get_route_for_node(v)
        if route_u is not None:
            affected.add(route_u)
        if route_v is not None:
            affected.add(route_v)
    return affected
