"""
SLC Policy Adapter.

Adapts the Soccer League Competition (SLC) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.slc import SLCConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.soccer_league_competition.params import SLCParams
from logic.src.policies.route_construction.meta_heuristics.soccer_league_competition.solver import SLCSolver


@RouteConstructorRegistry.register("slc")
class SLCPolicy(BaseRoutingPolicy):
    """
    SLC policy class.

    Visits bins using the Soccer League Competition algorithm.
    """

    def __init__(self, config: Optional[Union[SLCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SLCConfig

    def _get_config_key(self) -> str:
        return "slc"

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
        params = SLCParams(
            n_teams=int(values.get("n_teams", 5)),
            team_size=int(values.get("team_size", 4)),
            max_iterations=int(values.get("max_iterations", 50)),
            stagnation_limit=int(values.get("stagnation_limit", 5)),
            n_removal=int(values.get("n_removal", 1)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SLCSolver(
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
