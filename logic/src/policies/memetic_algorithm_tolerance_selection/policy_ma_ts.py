"""
Memetic Algorithm with Tolerance-based Selection (MA-TS) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import MemeticAlgorithmToleranceBasedSelectionParams
from .solver import MemeticAlgorithmToleranceBasedSelectionSolver


@PolicyRegistry.register("ma_ts")
class MemeticAlgorithmToleranceBasedSelectionPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Tolerance-based Selection policy class.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmToleranceBasedSelectionConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticAlgorithmToleranceBasedSelectionConfig

    def _get_config_key(self) -> str:
        return "ma_ts"

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
        params = MemeticAlgorithmToleranceBasedSelectionParams(
            population_size=values.get("population_size", 10),
            max_iterations=values.get("max_iterations", 100),
            tolerance_pct=values.get("tolerance_pct", 0.05),
            recombination_rate=values.get("recombination_rate", 0.6),
            perturbation_strength=values.get("perturbation_strength", 2),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = MemeticAlgorithmToleranceBasedSelectionSolver(
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
