"""
Step Counting Hill Climbing (SCHC) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.schc import SCHCConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import SCHCParams
from .solver import SCHCSolver


@PolicyRegistry.register("schc")
class StepCountingHillClimbingPolicy(BaseRoutingPolicy):
    """
    Adapter for the Step Counting Hill Climbing (SCHC) solver.
    """

    def __init__(self, config: Optional[Union[SCHCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SCHCConfig

    def _get_config_key(self) -> str:
        return "schc"

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
        params = SCHCParams(
            max_iterations=values.get("max_iterations", 1000),
            step_size=values.get("step_size", 100),
            time_limit=values.get("time_limit", 60.0),
            n_removal=values.get("n_removal", 2),
            n_llh=values.get("n_llh", 5),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )
        solver = SCHCSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )
        return solver.solve()
