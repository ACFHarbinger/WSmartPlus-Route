"""
Memetic Algorithm with Island Model (MA-IM) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ma_im import MemeticAlgorithmIslandModelConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import MemeticAlgorithmIslandModelParams
from .solver import MemeticAlgorithmIslandModelSolver


@PolicyRegistry.register("ma_im")
class MemeticAlgorithmIslandModelPolicy(BaseRoutingPolicy):
    """
    Memetic Algorithm Island Model policy class.
    """

    def __init__(self, config: Optional[Union[MemeticAlgorithmIslandModelConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MemeticAlgorithmIslandModelConfig

    def _get_config_key(self) -> str:
        return "ma_im"

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
        params = MemeticAlgorithmIslandModelParams(
            n_islands=values.get("n_islands", 5),
            island_size=values.get("island_size", 4),
            max_generations=values.get("max_generations", 50),
            stagnation_limit=values.get("stagnation_limit", 5),
            n_removal=values.get("n_removal", 1),
            local_search_iterations=values.get("local_search_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
        )

        solver = MemeticAlgorithmIslandModelSolver(
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
