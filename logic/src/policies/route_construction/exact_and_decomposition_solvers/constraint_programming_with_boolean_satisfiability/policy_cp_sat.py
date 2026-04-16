"""
Constraint Programming (CP-SAT) Policy Adapter.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.cp_sat import CPSATConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .cp_sat_engine import CPSATEngine


@RouteConstructorRegistry.register("cp_sat")
class CPSATPolicy(BaseRoutingPolicy):
    """
    Adapter for the Constraint Programming with Boolean Satisfiability (CP-SAT) policy using OR-Tools.
    Models the multi-day stochastic problem as an integrated CP optimization.
    """

    @classmethod
    def _config_class(cls):
        return CPSATConfig

    def _get_config_key(cls) -> str:
        return "cp_sat"

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
        """
        Satisfies the abstract base class, but CP-SAT uses the execute method.
        """
        return [], 0.0, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        Executes the CP-SAT solver.
        """
        # 1. State extraction
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0)

        cfg: CPSATConfig = self.config

        # 2. Instantiate and run Engine
        engine = CPSATEngine(config=cfg, distance_matrix=distance_matrix, initial_wastes=wastes, capacity=capacity)

        # 3. Solve and return Day 1 action
        route, expected_val = engine.solve()

        if not route or len(route) < 2:
            return [0, 0], 0.0, {"expected_value": expected_val, "stats": engine.stats}

        # Calculate actual cost of the extracted route for return
        actual_cost = 0.0
        for i in range(len(route) - 1):
            actual_cost += distance_matrix[route[i], route[i + 1]]

        return route, float(actual_cost), {"expected_value": expected_val, "stats": engine.stats}
