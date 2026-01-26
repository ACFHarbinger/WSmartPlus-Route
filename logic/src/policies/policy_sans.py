"""
SANS Policy Adapter.

Adapts the Simulated Annealing Neighborhood Search (SANS) logic
from LookAhead with VRPP-style statistical bin selection.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.look_ahead import policy_lookahead_sans
from logic.src.policies.single_vehicle import find_route, get_route_cost, local_search_2opt


@PolicyRegistry.register("sans")
class SANSPolicy(IPolicy):
    """
    SANS (Simulated Annealing Neighborhood Search) policy class.
    Uses statistical prediction for must-go bins and SA for routing.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the SANS policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        new_data = kwargs["new_data"]
        coords = kwargs["coords"]
        area = kwargs["area"]
        waste_type = kwargs["waste_type"]
        distancesC = kwargs["distancesC"]
        run_tsp = kwargs["run_tsp"]
        two_opt_max_iter = kwargs.get("two_opt_max_iter", 0)
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_sans_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0  # Default

        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means' attribute.")

        means = bins.means
        std = bins.std
        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        # Must-go bins (0-based indices as expected by policy_lookahead_sans internally?
        # Actually LookAheadPolicy passes it as-is, and policy_lookahead_sans does b+1)
        must_go_bins = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        if not must_go_bins:
            return [0, 0], 0.0, None

        # 2. Prepare Parameters and Values
        vehicle_capacity, R, B, C, E = load_area_and_waste_type_params(area, waste_type)

        sans_config = config.get("lookahead", {}).get("sans", {})
        values = {
            "R": R,
            "C": C,
            "E": E,
            "B": B,
            "vehicle_capacity": vehicle_capacity,
            "Omega": config.get("lookahead", {}).get("Omega", 0.1),
            "time_limit": sans_config.get("time_limit", 60),
            "perc_bins_can_overflow": sans_config.get("perc_bins_can_overflow", 0),
        }

        T_min = sans_config.get("T_min", 0.01)
        T_init = sans_config.get("T_init", 75)
        iterations_per_T = sans_config.get("iterations_per_T", 5000)
        alpha = sans_config.get("alpha", 0.95)
        params = (T_init, iterations_per_T, alpha, T_min)

        # 3. Prepare Data
        # Match LookAheadPolicy.execute transformation
        # new_data.loc[1:, "Stock"] = bins.c.astype("float32")
        # new_data.loc[1:, "Accum_Rate"] = bins.means.astype("float32")
        # Note: We should probably work on a copy to avoid side effects if this kwargs is shared
        new_data_copy = new_data.copy()
        new_data_copy.loc[1:, "Stock"] = bins.c.astype("float32")
        new_data_copy.loc[1:, "Accum_Rate"] = bins.means.astype("float32")

        # 4. Run SANS
        routes, _, _ = policy_lookahead_sans(new_data_copy, coords, distance_matrix, params, must_go_bins, values)

        tour = []
        if routes:
            routes = routes[0]  # Take first route
            tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
            if two_opt_max_iter > 0:
                tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)

        if not tour:
            tour = [0, 0]

        # Recalculate cost
        cost = get_route_cost(distance_matrix, tour)

        return tour, cost, None
