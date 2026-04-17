"""
Intensification Operators Package.

Intensification operators are search mechanisms designed to aggressively
exploit the immediate neighbourhood of a known high-quality solution —
"squeezing" the remaining inefficiency out to find the true local optimum
in the current region of the search space.

This package provides three tiers of intensification, each callable as a
standalone function on a ``List[List[int]]`` route plan:

Tier 1 — Heuristic Local Search (steepest descent)
----------------------------------------------------
Operators that methodically evaluate a dense neighbourhood and apply the
globally best improving move at each iteration, terminating at a strict
local minimum.

* **2-opt** (:func:`two_opt_steepest`, :func:`two_opt_steepest_profit`) —
  Reverses sub-sequences within each route; O(n²) per iteration.

* **Or-opt** (:func:`or_opt_steepest`, :func:`or_opt_steepest_profit`) —
  Relocates chains of 1, 2, or 3 consecutive nodes intra- and inter-route;
  covers the full Or-opt-3 neighbourhood in a single scan.

* **Node Exchange** (:func:`node_exchange_steepest`,
  :func:`node_exchange_steepest_profit`) — Steepest-descent pairwise node
  swap, both within and across routes, with capacity feasibility checking.

Tier 2 — Exact Algorithms (DP)
-------------------------------
Exact solvers applied to small, isolated sub-problems extracted from the
current plan.

* **DP Route Re-optimisation** (:func:`dp_route_reopt`,
  :func:`dp_route_reopt_profit`) — Runs the Held-Karp TSP dynamic
  programming algorithm on each short route (≤ max_nodes customers) to
  find the exact optimal node ordering.

Tier 3 — Matheuristics (Gurobi Sub-MIP)
-----------------------------------------
Use Gurobi to find proven-optimal solutions to restricted sub-problems.
Require Gurobi ≥ 11.0 with a valid licence.

* **Fix-and-Optimize** (:func:`fix_and_optimize`,
  :func:`fix_and_optimize_profit`) — Freezes the best routes and hands
  the remaining customers to a CVRP or VRPP sub-MIP for exact
  re-optimisation.

* **Set-Partitioning Polish** (:func:`set_partitioning_polish`,
  :func:`set_partitioning_polish_profit`) — Collects a pool of previously
  generated high-quality routes and solves the exact Set-Partitioning MIP
  to find the optimal route combination from that pool.

All operators follow the package-wide calling convention::

    improved_routes = operator(routes, dist_matrix, wastes, capacity, ...)

VRPP variants additionally accept ``R`` (revenue per unit waste) and ``C``
(cost per unit distance).  All operators return a new ``List[List[int]]``
and never mutate the input *routes* object.

Registry
--------
``INTENSIFICATION_OPERATORS`` maps string keys to their corresponding
functions for use in ALNS / ILS operator pools.  CVRP operators are keyed
directly; VRPP operators carry a ``_PROFIT`` suffix.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import (
    ...     two_opt_steepest, or_opt_steepest, dp_route_reopt, fix_and_optimize
    ... )
    >>> improved = two_opt_steepest(routes, dist_matrix, wastes, capacity)
    >>> improved = or_opt_steepest(routes, dist_matrix, wastes, capacity, C=0.5)
    >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity,
    ...                           max_nodes=15)
    >>> improved = fix_and_optimize(routes, dist_matrix, wastes, capacity,
    ...                             free_fraction=0.3, time_limit=30.0)
"""

from .dp_route_reopt import dp_route_reopt, dp_route_reopt_profit
from .fix_and_optimize import fix_and_optimize, fix_and_optimize_profit
from .set_partitioning_polish import (
    set_partitioning_polish,
    set_partitioning_polish_profit,
)

# ---------------------------------------------------------------------------
# Operator registry (CVRP and VRPP variants)
# ---------------------------------------------------------------------------

INTENSIFICATION_OPERATORS = {
    # Tier 1: Exact DP
    "DP_REOPT": dp_route_reopt,
    "DP_REOPT_PROFIT": dp_route_reopt_profit,
    # Tier 2: Matheuristic Sub-MIP (requires Gurobi)
    "FIX_OPT": fix_and_optimize,
    "FIX_OPT_PROFIT": fix_and_optimize_profit,
    "SP_POLISH": set_partitioning_polish,
    "SP_POLISH_PROFIT": set_partitioning_polish_profit,
}

INTENSIFICATION_NAMES = list(INTENSIFICATION_OPERATORS.keys())

__all__ = [
    "INTENSIFICATION_NAMES",
    "INTENSIFICATION_OPERATORS",
    # Tier 1
    "dp_route_reopt",
    "dp_route_reopt_profit",
    # Tier 2
    "fix_and_optimize",
    "fix_and_optimize_profit",
    "set_partitioning_polish",
    "set_partitioning_polish_profit",
]
