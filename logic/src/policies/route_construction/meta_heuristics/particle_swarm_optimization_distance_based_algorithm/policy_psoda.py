"""
Distance-Based PSO Policy Adapter.

Adapts the rigorous Distance-Based PSO implementation (replaces Firefly Algorithm).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import DistancePSOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import RouteConstructorRegistry

from .params import DistancePSOParams
from .solver import DistancePSOSolver


@RouteConstructorRegistry.register("psoda")
class DistancePSOPolicy(BaseRoutingPolicy):
    """
    Distance-Based Particle Swarm Optimization policy class.

    PSO with exponential distance decay. Replaces Firefly Algorithm.
    """

    def __init__(self, config: Optional[Union[DistancePSOConfig, Dict[str, Any]]] = None):
        """Initialize Distance-Based PSO policy with optional config.

        Args:
            config: DistancePSOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return DistancePSOConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "psoda"

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
        Run Distance-Based PSO solver.

        Returns:
            Tuple of (routes, profit, solver_cost)
        """
        params = DistancePSOParams(
            population_size=values.get("population_size", 20),
            max_iterations=values.get("max_iterations", 500),
            inertia_weight_start=values.get("inertia_weight_start", 0.9),
            inertia_weight_end=values.get("inertia_weight_end", 0.4),
            cognitive_coef=values.get("cognitive_coef", 2.0),
            social_coef=values.get("social_coef", 2.0),
            n_removal=values.get("n_removal", 3),
            velocity_to_mutation_rate=values.get("velocity_to_mutation_rate", 0.1),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            alpha_profit=values.get("alpha_profit", 1.0),
            beta_will=values.get("beta_will", 0.5),
            gamma_cost=values.get("gamma_cost", 0.3),
            vrpp=values.get("vrpp", False),
            profit_aware_operators=values.get("profit_aware_operators", True),
            seed=values.get("seed", 42),
        )

        solver = DistancePSOSolver(
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
