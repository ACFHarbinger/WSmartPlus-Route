"""
Dynamic Seed Hurdle Utility.

Provides the `_dynamic_seed_hurdle` function used by all profit-aware
insertion operators to determine whether it is worth opening a speculative
singleton route for a node.

The static threshold ``-0.5 × round_trip_cost × C`` used previously was
calibrated for an "average" instance state but ignores two factors that
determine how much subsidy a speculative route can realistically recoup:

1. **Pool density** ρ = |remaining optional nodes| / |total optional nodes|.
   When many nodes are still unassigned there are more potential co-riders
   to share the route's fixed cost; when few remain the window for synergy
   is small and the threshold should tighten.

2. **Expected Clarke-Wright synergy** E_s = mean CW savings of the candidate
   node with the remaining optional nodes:
   E_s = mean_v[ d(depot, u) + d(depot, v) − d(u, v) ] × C.
   High synergy (co-riders are geographically close) → larger subsidy is
   justified. Negative synergy (co-routing would cost more) → no subsidy.

Formula::

    hurdle = clamp( −ρ × max(E_s, 0),  [−new_cost × C, 0] )

Properties:
  - hurdle ≤ 0: never stricter than break-even.
  - hurdle ≥ −new_cost × C: never subsidises a full round-trip loss.
  - hurdle → 0 as ρ → 0: threshold tightens to break-even at the end of
    insertion.
  - hurdle → −E_s × C as ρ → 1: maximum permissiveness proportional to
    the expected co-rider synergy.

References:
    Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a
    central depot to a number of delivery points. OR, 12(4), 568-581.

Attributes:
    _dynamic_seed_hurdle: Compute a density-driven speculative seeding threshold.

Example:
    >>> from dynamic_seed_hurdle import _dynamic_seed_hurdle
    >>> hurdle = _dynamic_seed_hurdle(node=5, unassigned=[1,2,3,4,5],
    ...     mandatory_nodes_set=set(), dist_matrix=d, new_cost=10.0,
    ...     C=1.0, n_total_optional=10)
"""

from typing import Collection

import numpy as np


def _dynamic_seed_hurdle(
    node: int,
    unassigned: Collection[int],
    mandatory_nodes_set: set,
    dist_matrix: np.ndarray,
    new_cost: float,
    C: float,
    n_total_optional: int,
) -> float:
    """
    Compute a density-driven lower-bound threshold for speculative route seeding.

    A new singleton route for ``node`` is opened speculatively when::

        standalone_profit >= seed_hurdle

    The threshold adapts to the current insertion state using two factors:

    **Pool density** ρ ∈ (0, 1]:
        ρ = |optional_remaining| / n_total_optional
        Falls monotonically from 1 (start of insertion) toward 0 (end).
        As ρ falls the threshold tightens — fewer co-riders means less
        expected cost recovery.

    **Expected Clarke-Wright synergy** E_s (converted to cost units):
        E_s = mean_v[ d(0, node) + d(0, v) − d(node, v) ] × C  for v in optional_remaining
        Positive → merging with v saves distance; high E_s permits a larger
        subsidy because co-routing is likely profitable.
        Negative synergy (nodes on the opposite side of the depot) → E_s
        clamped to 0 so no subsidy is allowed.

    Formula::

        hurdle = clamp( −ρ × max(E_s, 0),  [−new_cost × C, 0] )

    Args:
        node: Candidate node for speculative singleton seeding.
        unassigned: All nodes not yet assigned at the point of evaluation.
            May include mandatory nodes (they are excluded from the density
            count) and the node itself (excluded by the v != node filter).
        mandatory_nodes_set: Mandatory nodes, excluded from the density
            count because they are always inserted and do not contribute
            to the optional co-rider pool.
        dist_matrix: Full distance matrix; index 0 is the depot.
        new_cost: d(depot, node) + d(node, depot) — round-trip cost for
            a singleton route serving only ``node``.
        C: Cost multiplier (currency per distance unit).
        n_total_optional: Total optional nodes in this instance
            (len(dist_matrix) - 1 - len(mandatory_nodes_set)).
            Used as the fixed denominator for ρ.

    Returns:
        float: Seed hurdle ≤ 0. The speculative route is opened when
        ``standalone_profit >= hurdle`` (or when the node is mandatory).
    """
    # Co-rider pool: optional nodes not yet decided, excluding current node.
    optional_remaining = [v for v in unassigned if v not in mandatory_nodes_set and v != node]
    n_remaining = len(optional_remaining)

    if n_remaining == 0 or n_total_optional == 0:
        # No potential co-riders remain — route must stand on its own profit.
        return 0.0

    # ρ ∈ (0, 1]: fraction of optional nodes still available as co-riders.
    density = n_remaining / n_total_optional

    # Clarke-Wright savings S(u, v) = d(0,u) + d(0,v) − d(u,v).
    # Positive: merging routes u and v saves total distance vs. dedicated routes.
    # Negative: u and v are on opposite sides — co-routing is counter-productive.
    d_0_u = float(dist_matrix[0, node])
    cw_savings = [d_0_u + float(dist_matrix[0, v]) - float(dist_matrix[node, v]) for v in optional_remaining]

    # Mean synergy as a conservative (non-over-optimistic) estimate.
    # Clamped to 0: we never impose a negative hurdle offset for anti-synergistic pools.
    mean_synergy = float(np.mean(cw_savings))
    expected_synergy_cost = max(mean_synergy, 0.0) * C

    raw_hurdle = -density * expected_synergy_cost

    # Clamp to [−new_cost × C, 0]:
    #   Upper bound 0: never stricter than break-even (profit check covers that).
    #   Lower bound −new_cost × C: never subsidise a full round-trip loss.
    return max(-new_cost * C, min(0.0, raw_hurdle))
