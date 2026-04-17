"""
SolutionContext — Current-day routing solution.

Stores the routes actually executed today (one per vehicle) together with
per-vehicle profit and cost breakdowns and aggregate totals.
The multi-day plan beyond today is stored in MultiDayContext.full_plan_snapshot
and is NOT duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class SolutionContext:
    """
    Today's routing solution, as returned by _run_multi_period_solver.

    Attributes:
        routes:        List of routes for today. Each route is a List[int] of
                       node IDs for one vehicle (depot excluded; e.g. [3, 7, 2]).
                       Empty list [] means that vehicle is idle.
                       Typically len(routes) == n_vehicles or 1 for single-vehicle.
        profits:       Per-route daily profit (r_w·Σw_i − c_km·dist).
                       len(profits) == len(routes).
        costs:         Per-route travel cost (c_km · dist only, no revenue).
                       len(costs) == len(routes).
        total_profit:  Sum of profits across all today's routes.
        total_cost:    Sum of costs across all today's routes.
        metadata:      Optional solver-specific telemetry (e.g. MIP gap,
                       nodes explored, LR iterations).
    """

    routes: List[List[int]]
    profits: List[float]
    costs: List[float]
    total_profit: float
    total_cost: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SolutionContext":
        """Return an empty (no-visit) solution."""
        return cls(routes=[], profits=[], costs=[], total_profit=0.0, total_cost=0.0)

    @classmethod
    def from_single_route(
        cls,
        route: List[int],
        profit: float,
        cost: float,
        metadata: Optional[dict] = None,
    ) -> "SolutionContext":
        """Convenience constructor for single-vehicle results."""
        return cls(
            routes=[route],
            profits=[profit],
            costs=[cost],
            total_profit=profit,
            total_cost=cost,
            metadata=metadata or {},
        )

    def to_flat_tour(self) -> List[int]:
        """
        Flatten all routes into a single depot-delimited tour.

        Produces [0, r1_n1, r1_n2, ..., 0, r2_n1, ...., 0] suitable for
        passing back to the simulator's single-tour interface.
        """
        flat = []
        for route in self.routes:
            if route:
                flat.append(0)
                flat.extend(route)
        flat.append(0)
        return flat if flat else [0, 0]
