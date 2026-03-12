"""
Continuous Local Search Policy Adapter.

Adapts the rigorous Continuous Local Search implementation (replaces SCA).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ContinuousLocalSearchConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.continuous_local_search import ContinuousLocalSearchParams, ContinuousLocalSearchSolver


@PolicyRegistry.register("continuous_ls")
class ContinuousLocalSearchPolicy(BaseRoutingPolicy):
    """
    Continuous Local Search policy class.

    Gradient-free search with trigonometric perturbations. Replaces Sine Cosine Algorithm.
    """

    def __init__(self, config: Optional[Union[ContinuousLocalSearchConfig, Dict[str, Any]]] = None):
        """Initialize Continuous Local Search policy with optional config.

        Args:
            config: ContinuousLocalSearchConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ContinuousLocalSearchConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "continuous_ls"

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
        Run Continuous Local Search solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = ContinuousLocalSearchParams(
            population_size=values.get("population_size", 30),
            max_step_size=values.get("max_step_size", 2.0),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = ContinuousLocalSearchSolver(
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
