"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .adapters import PolicyRegistry
from .adaptive_large_neighborhood_search import run_alns
from .base_routing_policy import BaseRoutingPolicy


@PolicyRegistry.register("alns")
class ALNSPolicy(BaseRoutingPolicy):
    """
    ALNS policy class.

    Visits pre-selected 'must_go' bins using Adaptive Large Neighborhood Search.
    """

    def _get_config_key(self) -> str:
        """Return config key for ALNS."""
        return "alns"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """
        Run ALNS solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        routes, _, solver_cost = run_alns(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
        )
        return routes, solver_cost
