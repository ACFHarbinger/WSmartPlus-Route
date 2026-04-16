"""
GLS (Guided Local Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gls import GLSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import GLSParams
from .solver import GLSSolver


@RouteConstructorRegistry.register("gls")
class GLSPolicy(BaseRoutingPolicy):
    """Guided Large Neighborhood Search (G-LNS) policy class."""

    def __init__(self, config: Optional[Union[GLSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GLSConfig

    def _get_config_key(self) -> str:
        return "gls"

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
        params = GLSParams(
            lambda_param=float(values.get("lambda_param", 1.0)),
            alpha_param=float(values.get("alpha_param", 0.3)),
            penalty_cycles=int(values.get("penalty_cycles", 1000)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 6)),
            inner_iterations=int(values.get("inner_iterations", 100)),
            fls_coupling_prob=float(values.get("fls_coupling_prob", 0.8)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = GLSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
