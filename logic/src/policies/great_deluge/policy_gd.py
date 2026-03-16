"""
Great Deluge (GD) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gd import GDConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import GDParams
from .solver import GDSolver


@PolicyRegistry.register("gd")
class GreatDelugePolicy(BaseRoutingPolicy):
    """
    Adapter for the Great Deluge (GD) solver.
    """

    def __init__(self, config: Optional[Union[GDConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GDConfig

    def _get_config_key(self) -> str:
        return "gd"

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
        params = GDParams(
            max_iterations=values.get("max_iterations", 1000),
            target_fitness_multiplier=values.get("target_fitness_multiplier", 1.1),
            time_limit=values.get("time_limit", 60.0),
            n_removal=values.get("n_removal", 2),
            n_llh=values.get("n_llh", 5),
        )
        solver = GDSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
            seed=values.get("seed"),
        )
        return solver.solve()
