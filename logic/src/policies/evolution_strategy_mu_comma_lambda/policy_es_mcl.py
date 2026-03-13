"""
(μ,λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ,λ)-ES implementation (replaces Artificial Bee Colony).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuCommaLambdaESConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.evolution_strategy_mu_comma_lambda import MuCommaLambdaESParams, MuCommaLambdaESSolver


@PolicyRegistry.register("es_mcl")
class MuCommaLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ,λ) Evolution Strategy policy class.

    Multi-phase ES with random restart. Replaces Artificial Bee Colony.
    """

    def __init__(self, config: Optional[Union[MuCommaLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ,λ)-ES policy with optional config.

        Args:
            config: MuCommaLambdaESConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MuCommaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "es_mcl"

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
        Run (μ,λ)-ES solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = MuCommaLambdaESParams(
            population_size=values.get("population_size", 20),
            offspring_per_parent=values.get("offspring_per_parent", 1),
            n_removal=values.get("n_removal", 3),
            stagnation_limit=values.get("stagnation_limit", 10),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = MuCommaLambdaESSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
