"""
(μ+λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ+λ)-ES implementation (replaces Harmony Search).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuPlusLambdaESConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.evolution_strategy_mu_plus_lambda import MuPlusLambdaESParams, MuPlusLambdaESSolver


@PolicyRegistry.register("es_mpl")
class MuPlusLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ+λ) Evolution Strategy policy class.

    Canonical ES with recombination and mutation. Replaces Harmony Search.
    """

    def __init__(self, config: Optional[Union[MuPlusLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ+λ)-ES policy with optional config.

        Args:
            config: MuPlusLambdaESConfig dataclass, raw dict from YAML, or None.
        """
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
            population_size=values.get("population_size", 10),
            offspring_size=values.get("offspring_size", 5),
            recombination_rate=values.get("recombination_rate", 0.95),
            mutation_rate=values.get("mutation_rate", 0.3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = MuPlusLambdaESSolver(
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
