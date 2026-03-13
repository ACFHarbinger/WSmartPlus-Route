"""
Memetic Algorithm with Dual Population (MA-DP) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_dp import MemeticAlgorithmDualPopulationConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import MemeticAlgorithmDualPopulationParams
from .solver import MemeticAlgorithmDualPopulationSolver


@PolicyRegistry.register("ma_dp")
class MemeticAlgorithmDualPopulationPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Dual Population policy class.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmDualPopulationConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticAlgorithmDualPopulationConfig

    def _get_config_key(self) -> str:
        return "ma_dp"

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
        params = MemeticAlgorithmDualPopulationParams(
            population_size=values.get("population_size", 30),
            max_iterations=values.get("max_iterations", 200),
            diversity_injection_rate=values.get("diversity_injection_rate", 0.2),
            elite_learning_weights=values.get("elite_learning_weights"),
            elite_count=values.get("elite_count", 3),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 300.0),
        )

        solver = MemeticAlgorithmDualPopulationSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
            seed=values.get("seed"),
        )

        return solver.solve()
