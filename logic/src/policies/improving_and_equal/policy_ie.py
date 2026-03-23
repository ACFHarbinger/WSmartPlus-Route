"""
Improving and Equal (IE) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ie import IEConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import IEParams
from .solver import IESolver


@PolicyRegistry.register("ie")
class ImprovingAndEqualPolicy(BaseRoutingPolicy):
    """
    Adapter for the Improving and Equal (IE) solver.
    """

    def __init__(self, config: Optional[Union[IEConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return IEConfig

    def _get_config_key(self) -> str:
        return "ie"

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
        params = IEParams(
            max_iterations=int(values.get("max_iterations", 1000)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=int(values.get("seed", 42)),
            vrpp=bool(values.get("vrpp", True)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
        )
        solver = IESolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )
        return solver.solve()
