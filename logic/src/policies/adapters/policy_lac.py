"""
LAC Policy Adapter (Look-ahead Algorithm for Collection).

Uses look-ahead optimization for route planning.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.constants.optimization import (
    DEFAULT_COMBINATION,
    DEFAULT_SHIFT_DURATION,
    DEFAULT_TIME_LIMIT,
    DEFAULT_V_VALUE,
)
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import create_points
from logic.src.policies.simulated_annealing_neighborhood_search.refinement.route_search import find_solutions
from logic.src.policies.single_vehicle import get_route_cost

from .factory import PolicyRegistry


@PolicyRegistry.register("lac")
class LACPolicy(BaseRoutingPolicy):
    """
    LAC policy class.

    Uses look-ahead optimization with simulated annealing for route construction.
    """

    def _get_config_key(self) -> str:
        """Return config key for LAC."""
        return "lac"

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
        """Not used - LAC requires specialized execute()."""
        return [[]], 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LAC policy.

        Uses specialized look-ahead solution finding.
        """
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        new_data = kwargs["new_data"]  # Expecting the dataframe
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        config = kwargs.get("config", {})
        lac_config = config.get("lac", {})

        # Load area parameters
        Q, R, B, C = self._load_area_params(area, waste_type, config)
        V = lac_config.get("V", DEFAULT_V_VALUE)

        values = {
            "Q": Q,
            "R": R,
            "B": B,
            "C": C,
            "V": V,
            "shift_duration": DEFAULT_SHIFT_DURATION,
            "perc_bins_can_overflow": 0,
        }
        values.update(lac_config)

        # must_go bins are 0-based, find_solutions expects 1-based
        must_go_1 = [b + 1 for b in must_go]

        points = create_points(new_data, coords)

        combination = lac_config.get("combination", DEFAULT_COMBINATION)

        try:
            res, _, _ = find_solutions(
                new_data,
                coords,
                distance_matrix,
                combination,
                must_go_1,
                values,
                bins.n,
                points,
                time_limit=lac_config.get("time_limit", DEFAULT_TIME_LIMIT),
            )
        except Exception:
            return [0, 0], 0.0, None

        tour = res[0] if res else [0, 0]
        cost = get_route_cost(distance_matrix, tour)

        return tour, cost, None
