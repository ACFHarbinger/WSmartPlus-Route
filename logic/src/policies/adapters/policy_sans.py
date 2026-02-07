"""
SANS Policy Adapter (Simulated Annealing).

Uses Simulated Annealing for route optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logic.src.configs.policies import SANSConfig
from logic.src.pipeline.simulations.processor import convert_to_dict
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.simulated_annealing_neighborhood_search import (
    improved_simulated_annealing,
)
from logic.src.policies.simulated_annealing_neighborhood_search.common.solution_initialization import (
    compute_initial_solution,
)

from .factory import PolicyRegistry


@PolicyRegistry.register("sans")
class SANSPolicy(BaseRoutingPolicy):
    """
    Simulated Annealing policy class.

    Uses SA optimization with custom initialization and must-go enforcement.
    """

    def __init__(self, config: Optional[SANSConfig] = None):
        """Initialize SANS policy with optional config.

        Args:
            config: Optional SANSConfig dataclass with solver parameters.
        """
        super().__init__(config)

    def _get_config_key(self) -> str:
        """Return config key for SANS."""
        return "sans"

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
        """Not used - SANS requires specialized execute()."""
        return [[]], 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the SANS policy.

        Uses specialized data preparation for simulated annealing.
        """
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        config = kwargs.get("config", {})
        sans_config = config.get("sans", {})

        # Load area parameters
        Q, R, C, values = self._load_area_params(area, waste_type, config)

        # Load B and V defaults which are not returned by base
        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        _, _, B_def, _, V_def = load_area_and_waste_type_params(area, waste_type)

        B = values.get("B", B_def)
        V = values.get("V", V_def)

        # Prepare data for SANS logic
        data = pd.DataFrame(
            {
                "ID": np.arange(1, bins.n + 1),
                "#bin": np.arange(1, bins.n + 1),
                "Stock": bins.c.astype("float32"),
                "Accum_Rate": bins.means.astype("float32"),
            }
        )
        # Add depot
        depot_row = pd.DataFrame([{"#bin": 0, "Stock": 0.0, "Accum_Rate": 0.0}])
        data = pd.concat([depot_row, data], ignore_index=True)

        coords_dict = convert_to_dict(coords)
        id_to_index = {i: i for i in range(len(distance_matrix))}

        initial_routes = compute_initial_solution(data, coords_dict, distance_matrix, Q, id_to_index)
        current_route = initial_routes[0] if initial_routes else [0, 0]

        # Ensure must_go bins are in the route
        must_go_1 = list(must_go)
        route_set = set(current_route)
        for b in must_go_1:
            if b not in route_set:
                current_route.insert(1, b)

        params = (
            sans_config.get("T_init", 75),
            sans_config.get("iterations_per_T", 5000),
            sans_config.get("alpha", 0.95),
            sans_config.get("T_min", 0.01),
        )

        optimized_routes, best_profit, last_distance, _, _ = improved_simulated_annealing(
            [current_route],
            distance_matrix,
            values.get("time_limit", 60),
            id_to_index,
            data,
            Q,
            *params,
            R,
            V,
            B,
            C,
            must_go_1,
            perc_bins_can_overflow=sans_config.get("perc_bins_can_overflow", 0.0),
            volume=V,
            density_val=B,
            max_vehicles=1,
        )

        tour = optimized_routes[0] if optimized_routes else [0, 0]
        return tour, last_distance, None
