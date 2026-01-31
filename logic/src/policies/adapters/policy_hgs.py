"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..base_routing_policy import BaseRoutingPolicy
from ..hybrid_genetic_search import run_hgs
from .factory import PolicyRegistry


@PolicyRegistry.register("hgs")
class HGSPolicy(BaseRoutingPolicy):
    """
    Hybrid Genetic Search policy class.

    Visits pre-selected 'must_go' bins using evolutionary optimization.
    """

    def _get_config_key(self) -> str:
        """Return config key for HGS."""
        return "hgs"

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
        Run HGS solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        routes, _, solver_cost = run_hgs(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
        )
        return routes, solver_cost
