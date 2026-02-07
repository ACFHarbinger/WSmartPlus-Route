"""
ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies import ACOConfig
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.ant_colony_optimization.k_sparse_aco import run_k_sparse_aco

from .factory import PolicyRegistry


@PolicyRegistry.register("aco")
class ACOPolicy(BaseRoutingPolicy):
    """
    K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.
    """

    def __init__(self, config: Optional[ACOConfig] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: Optional ACOConfig dataclass with solver parameters.
        """
        super().__init__(config)

    def _get_config_key(self) -> str:
        """Return config key for ACO."""
        return "aco"

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
        Run K-Sparse ACO solver.

        Returns:
            Tuple of (routes, solver_cost)
        """
        routes, _, solver_cost = run_k_sparse_aco(
            sub_dist_matrix,
            sub_demands,
            capacity,
            revenue,
            cost_unit,
            values,
        )
        return routes, solver_cost
