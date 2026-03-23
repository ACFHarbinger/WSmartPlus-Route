"""
Only Improving (OI) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.oi import OIConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import OIParams
from .solver import OISolver


@PolicyRegistry.register("oi")
class OnlyImprovingPolicy(BaseRoutingPolicy):
    """
    Adapter for the Only Improving (OI) solver.
    """

    def __init__(self, config: Optional[Union[OIConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return OIConfig

    def _get_config_key(self) -> str:
        return "oi"

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
        params = OIParams(
            max_iterations=values.get("max_iterations", 1000),
            n_removal=values.get("n_removal", 2),
            n_llh=values.get("n_llh", 5),
            time_limit=values.get("time_limit", 60.0),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )
        solver = OISolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )
        return solver.solve()
