"""
Simulated Annealing Policy Adapter.

Wraps the core SA meta-heuristic into the standard BaseRoutingPolicy interface
to permit unified execution by the factory manager.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import SAParams
from .solver import SASolver


@RouteConstructorRegistry.register("sa")
class SAPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing (SA) policy adapter.

    Instantiates the thermodynamic solver and executes the Markov chain
    search over the specified routing graph.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        # Using dictionary configurations internally, avoiding external dataclass dependencies
        return None

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
        # 1. Parse parameters
        params = SAParams.from_config(values)

        # 2. Instantiate core solver
        solver = SASolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Optimize and return
        best_routes, best_profit, best_cost = solver.solve(initial_solution=None)

        return best_routes, best_profit, best_cost
