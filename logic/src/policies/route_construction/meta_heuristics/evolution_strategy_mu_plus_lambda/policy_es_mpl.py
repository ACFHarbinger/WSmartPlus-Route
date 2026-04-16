"""
(μ+λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ+λ)-ES implementation into the overarching policy registry.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuPlusLambdaESConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MuPlusLambdaESParams
from .solver import MuPlusLambdaESSolver


@RouteConstructorRegistry.register("es_mpl")
class MuPlusLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ+λ) Evolution Strategy policy class.

    Executes a steady-state evolutionary algorithm with strong elitism.
    """

    def __init__(self, config: Optional[Union[MuPlusLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ+λ)-ES policy with optional config."""
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MuPlusLambdaESConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "es_mpl"

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
        Run (μ+λ)-ES solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = MuPlusLambdaESParams(
            mu=values.get("mu", 10),
            lambda_=values.get("lambda_", 5),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = MuPlusLambdaESSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
