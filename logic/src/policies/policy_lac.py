"""
LAC Policy Adapter (Look-ahead Algorithm for Collection).
"""
from typing import Any, List, Tuple

from .adapters import IPolicy, PolicyRegistry
from .look_ahead_aux.routes import create_points
from .look_ahead_aux.solutions import find_solutions


@PolicyRegistry.register("lac")
class LACPolicy(IPolicy):
    """
    LAC policy class.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LAC policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        coords = kwargs["coords"]
        new_data = kwargs["new_data"]  # Expecting the dataframe
        area = kwargs["area"]
        waste_type = kwargs["waste_type"]
        config = kwargs.get("config", {})
        lac_config = config.get("lac", {})

        from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

        Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
        values = {"Q": Q, "R": R, "B": B, "C": C, "V": V, "shift_duration": 390, "perc_bins_can_overflow": 0}
        values.update(lac_config)

        # must_go bins are 0-based, find_solutions expects?
        # Original code: must_go_bins = [b + 1 for b in must_go]
        # Checking find_solutions in look_ahead.py usage
        must_go_1 = [b + 1 for b in must_go]

        points = create_points(new_data, coords)
        # find_solutions expects data normalized?
        # new_data.loc[1:, "Stock"] = (bins.c/100).astype("float32")

        combination = lac_config.get("combination", [500, 75, 0.95, 0, 0.095, 0, 0])

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
                time_limit=lac_config.get("time_limit", 600),
            )
        except Exception:
            return [0, 0], 0.0, None

        tour = res[0] if res else [0, 0]

        from .single_vehicle import get_route_cost

        return tour, get_route_cost(distance_matrix, tour), None
