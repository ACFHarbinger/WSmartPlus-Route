"""
HGS-ALNS Hybrid Policy Adapter.

Adapts the Hybrid HGS-ALNS solver to the common simulator policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies import HGSALNSConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.hgs_alns import HGSALNSSolver
from logic.src.policies.hybrid_genetic_search.params import HGSParams

from .factory import PolicyRegistry


@PolicyRegistry.register("hgs_alns")
class HGSALNSPolicy(BaseRoutingPolicy):
    """
    Hybrid HGS-ALNS policy class for the simulator.

    Uses ALNS for the intensive education phase of HGS.
    """

    def __init__(self, config: Optional[HGSALNSConfig] = None):
        """Initialize HGS-ALNS policy with optional config.

        Args:
            config: Optional HGSALNSConfig dataclass with solver parameters.
        """
        super().__init__(config)

    def _get_config_key(self) -> str:
        """Return config key for HGS-ALNS hybrid."""
        return "hgs_alns"

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
        Run HGS-ALNS hybrid solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        # Get HGS params, falling back to values dict
        params = HGSParams(
            time_limit=values.get("time_limit", 10),
            population_size=values.get("population_size", 50),
            elite_size=values.get("elite_size", 10),
            mutation_rate=values.get("mutation_rate", 0.2),
            max_vehicles=values.get("max_vehicles", 0),
        )

        alns_iter = values.get("alns_education_iterations", 50)

        solver = HGSALNSSolver(
            dist_matrix=sub_dist_matrix,
            demands=sub_demands,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            alns_education_iterations=alns_iter,
        )

        routes, _, solver_cost = solver.solve()
        return routes, solver_cost
