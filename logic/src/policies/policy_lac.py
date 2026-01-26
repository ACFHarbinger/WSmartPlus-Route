"""
LAC Policy Adapter.

Adapts the LookAhead logic with specific overrides (shift_duration=390, perc_bins_can_overflow=0)
and VRPP-style statistical bin selection.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.look_ahead_aux.routes import create_points
from logic.src.policies.look_ahead_aux.solutions import find_solutions
from logic.src.policies.single_vehicle import find_route, get_route_cost, local_search_2opt


@PolicyRegistry.register("policy_lac")
class LACPolicy(IPolicy):
    """
    LAC policy class.
    Executes LookAhead solution search with specific parameter overrides.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LAC policy.
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
        graph_size = kwargs["graph_size"]
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_lac_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0

        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means' attribute.")

        means = bins.means
        std = bins.std
        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        must_go_bins = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        if not must_go_bins:
            return [0, 0], 0.0, None

        # 2. Prepare Parameters and Values
        vehicle_capacity, R, B, C, E = load_area_and_waste_type_params(area, waste_type)

        last_minute_config = config.get("lookahead", {})
        values = {
            "R": R,
            "C": C,
            "E": E,
            "B": B,
            "vehicle_capacity": vehicle_capacity,
            "Omega": last_minute_config.get("Omega", 0.1),
            "shift_duration": 390,
            "perc_bins_can_overflow": 0,
        }

        # Determine chosen_combination (look_ahead config char)
        # In LookAheadPolicy it does policy[policy.find("ahead_") + len("ahead_")]
        # For policy_lac, we might need a default or derive from name.
        # User snippet used 'chosen_combination'.
        # I'll default to 'a' or look for it in the name.
        look_ahead_config_char = "a"
        if "ahead_" in policy:
            look_ahead_config_char = policy[policy.find("ahead_") + len("ahead_")]

        possible_configurations = {
            "a": [500, 75, 0.95, 0, 0.095, 0, 0],
            "b": [2000, 75, 0.7, 0, 0.095, 0, 0],
        }
        # Override from config if present
        if look_ahead_config_char in last_minute_config:
            possible_configurations[look_ahead_config_char] = last_minute_config[look_ahead_config_char]

        chosen_combination = possible_configurations.get(look_ahead_config_char, possible_configurations["a"])

        # 3. Prepare Data
        new_data_copy = new_data.copy()
        points = create_points(new_data_copy, coords)
        new_data_copy.loc[1 : graph_size + 1, "Stock"] = (bins.c / 100).astype("float32")
        new_data_copy.loc[1 : graph_size + 1, "Accum_Rate"] = (bins.means / 100).astype("float32")

        # 4. Find Solutions (Retry Logic)
        routes = None
        try:
            routes, _, _ = find_solutions(
                new_data_copy,
                coords,
                distance_matrix,
                chosen_combination,
                must_go_bins,
                values,
                graph_size,
                points,
                time_limit=600,
            )
        except Exception:
            try:
                routes, _, _ = find_solutions(
                    new_data_copy,
                    coords,
                    distance_matrix,
                    chosen_combination,
                    must_go_bins,
                    values,
                    graph_size,
                    points,
                    time_limit=3600,
                )
            except Exception:
                routes = None

        tour = []
        if routes:
            routes = routes[0]
            tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
            if two_opt_max_iter > 0:
                tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)

        if not tour:
            tour = [0, 0]

        # Recalculate cost
        cost = get_route_cost(distance_matrix, tour)

        return tour, cost, None
