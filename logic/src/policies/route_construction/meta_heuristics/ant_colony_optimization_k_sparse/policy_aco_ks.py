"""
ACO Policy Adapter.

Adapts the K-Sparse Ant Colony Optimization solver to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import KSparseACOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry

from .params import KSACOParams
from .solver import KSparseACOSolver


@RouteConstructorRegistry.register("aco_ks")
class ACOPolicy(BaseRoutingPolicy):
    """
    K-Sparse Ant Colony Optimization policy class.

    Uses ACS with sparse pheromone matrix for efficient VRP solving.
    """

    def __init__(self, config: Optional[Union[KSparseACOConfig, Dict[str, Any]]] = None):
        """Initialize ACO policy with optional config.

        Args:
            config: KSparseACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return KSparseACOConfig

    def _get_config_key(self) -> str:
        """Return config key for ACO."""
        return "aco"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run K-Sparse ACO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = KSACOParams(
            n_ants=values.get("n_ants", 10),
            k_sparse=values.get("k_sparse", 15),
            alpha=values.get("alpha", 1.0),
            beta=values.get("beta", 2.0),
            rho=values.get("rho", 0.1),
            q0=values.get("q0", 0.9),
            tau_0=values.get("tau_0"),
            tau_min=values.get("tau_min", 0.001),
            tau_max=values.get("tau_max", 10.0),
            max_iterations=values.get("max_iterations", 100),
            time_limit=values.get("time_limit", 30.0),
            local_search=values.get("local_search", True),
            local_search_iterations=values.get("local_search_iterations", 500),
            elitist_weight=values.get("elitist_weight", 1.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = KSparseACOSolver(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params, mandatory_nodes)
        return solver.solve()
