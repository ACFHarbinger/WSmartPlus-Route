from typing import Any, Dict, List, Tuple

import numpy as np
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.slack_induction_by_string_removal import SISRParams, SISRSolver

from .factory import PolicyRegistry


@PolicyRegistry.register("sisr")
class SISRPolicy(BaseRoutingPolicy):
    """
    Policy adapter for the SISR metaheuristic.
    """

    def _get_config_key(self) -> str:
        return "sisr"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        params = SISRParams(
            time_limit=values.get("time_limit", 10.0),
            max_iterations=values.get("max_iterations", 1000),
            start_temp=values.get("start_temp", 100.0),
            cooling_rate=values.get("cooling_rate", 0.995),
            max_string_len=values.get("max_string_len", 10),
            avg_string_len=values.get("avg_string_len", 3.0),
            blink_rate=values.get("blink_rate", 0.01),
            destroy_ratio=values.get("destroy_ratio", 0.2),
        )

        solver = SISRSolver(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, params)
        routes, profit, cost = solver.solve()
        return routes, cost
