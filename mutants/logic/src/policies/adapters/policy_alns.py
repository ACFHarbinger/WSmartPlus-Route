"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from logic.src.configs.policies import ALNSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.adaptive_large_neighborhood_search.alns import run_alns

from .factory import PolicyRegistry


@PolicyRegistry.register("alns")
class ALNSPolicy(BaseRoutingPolicy):
    """
    ALNS policy class.

    Visits pre-selected 'must_go' bins using Adaptive Large Neighborhood Search.
    """

    def __init__(self, config: Optional[ALNSConfig] = None):
        """Initialize ALNS policy with optional config.

        Args:
            config: Optional ALNSConfig dataclass with solver parameters.
        """
        super().__init__(config)

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
