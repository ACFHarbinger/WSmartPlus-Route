"""
Threshold Accepting (TA) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ta import TAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import TAParams
from .solver import TASolver


@PolicyRegistry.register("ta")
class ThresholdAcceptingPolicy(BaseRoutingPolicy):
    """
    Adapter for the Threshold Accepting (TA) solver.
    """

    def __init__(self, config: Optional[Union[TAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return TAConfig

    def _get_config_key(self) -> str:
        return "ta"

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
        params = TAParams(
            max_iterations=values.get("max_iterations", 1000),
            initial_threshold=values.get("initial_threshold", 100.0),
            time_limit=values.get("time_limit", 60.0),
            n_removal=values.get("n_removal", 2),
            n_llh=values.get("n_llh", 5),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )
        solver = TASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )
        return solver.solve()
