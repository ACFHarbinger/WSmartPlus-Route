"""
Distance-Based PSO Policy Adapter.

Adapts the rigorous Distance-Based PSO implementation (replaces Firefly Algorithm).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import DistancePSOConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.particle_swarm_optimization_distance import DistancePSOParams, DistancePSOSolver


@PolicyRegistry.register("distance_pso")
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
        return "distance_pso"

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
            initial_attraction=values.get("initial_attraction", 1.0),
            distance_decay=values.get("distance_decay", 0.01),
            exploration_rate=values.get("exploration_rate", 0.1),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            alpha_profit=values.get("alpha_profit", 1.0),
            beta_will=values.get("beta_will", 0.5),
            gamma_cost=values.get("gamma_cost", 0.3),
        )

        solver = DistancePSOSolver(
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
