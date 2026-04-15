"""
GENIUS (GENI + US) Policy Adapter.

Reference:
    Gendreau, M., Hertz, A., & Laporte, G. (1992).
    "New Insertion and Postoptimization Procedures for the Traveling Salesman Problem"
    Operations Research, 40(6), 1086-1094.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.genius import GENIUSConfig
from logic.src.policies.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.base.factory import PolicyRegistry

from .params import GENIUSParams
from .solver import GENIUSSolver


@PolicyRegistry.register("genius")
class GENIUSPolicy(BaseRoutingPolicy):
    """GENIUS (GENI + US) policy class."""

    def __init__(self, config: Optional[Union[GENIUSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GENIUSConfig

    def _get_config_key(self) -> str:
        return "genius"

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
        params = GENIUSParams(
            neighborhood_size=int(values.get("neighborhood_size", 5)),
            unstring_type=int(values.get("unstring_type", 1)),
            string_type=int(values.get("string_type", 1)),
            n_iterations=int(values.get("n_iterations", 1)),
            random_us_sampling=bool(values.get("random_us_sampling", False)),
            vrpp=bool(values.get("vrpp", False)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
        )

        solver = GENIUSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
