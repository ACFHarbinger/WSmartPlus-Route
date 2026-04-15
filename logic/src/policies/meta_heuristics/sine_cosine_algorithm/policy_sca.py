"""
SCA Policy Adapter.

Adapts the Sine Cosine Algorithm (SCA) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.sca import SCAConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry
from logic.src.policies.meta_heuristics.sine_cosine_algorithm.params import SCAParams
from logic.src.policies.meta_heuristics.sine_cosine_algorithm.solver import SCASolver


@PolicyRegistry.register("sca")
class SCAPolicy(BaseRoutingPolicy):
    """
    SCA policy class.

    Visits bins using the Sine Cosine Algorithm.
    """

    def __init__(self, config: Optional[Union[SCAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SCAConfig

    def _get_config_key(self) -> str:
        return "sca"

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
        params = SCAParams(
            pop_size=int(values.get("pop_size", 20)),
            a_max=float(values.get("a_max", 2.0)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SCASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
