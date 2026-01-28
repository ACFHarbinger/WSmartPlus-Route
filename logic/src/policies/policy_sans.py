"""
SANS Policy Adapter (Simulated Annealing).
"""
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from logic.src.pipeline.simulations.processor import convert_to_dict

from .adapters import IPolicy, PolicyRegistry
from .look_ahead_aux import compute_initial_solution, improved_simulated_annealing


@PolicyRegistry.register("sans")
class SANSPolicy(IPolicy):
    """
    Simulated Annealing policy class.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the SANS policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        area = kwargs["area"]
        waste_type = kwargs["waste_type"]
        config = kwargs.get("config", {})
        sans_config = config.get("sans", {})

        from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

        Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)

        # Prepare data for SANS logic
        # Create DataFrame directly instead of using create_dataframe_from_matrix which might expect 2D
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
        must_go_1 = list(must_go)  # must_go is 1-based
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
            sans_config.get("time_limit", 60),
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
