"""
VRPP Policy Wrapper.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from logic.src.policies.adapters.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.adapters.factory import PolicyRegistry
from logic.src.policies.vehicle_routing_problem_with_profits.interface import run_vrpp_optimizer
from logic.src.utils.data.data_utils import load_area_and_waste_type_params


@PolicyRegistry.register("vrpp")
class VRPPPolicy(BaseRoutingPolicy):
    """
    Agnostic VRPP Policy adapter.
    Delegates to run_vrpp_optimizer.
    """

    def _get_config_key(self) -> str:
        return "vrpp"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ):
        """Not used - VRPP requires specialized execute()."""
        pass

    def execute(self, **kwargs: Any):
        """
        Execute the VRPP policy.
        """
        # Extract arguments
        area = kwargs.get("area")
        waste_type = kwargs.get("waste_type")
        bins = kwargs.get("bins")  # waste amounts
        distance_matrix = kwargs.get("distance_matrix")
        binsids = kwargs.get("binsids")
        must_go = kwargs.get("must_go", [])

        # Load params (reusing logic from original file)
        # Assuming defaults if not present
        if area and waste_type:
            Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
            values = {"Q": Q, "R": R, "B": B, "C": C, "V": V}
        else:
            # Fallback
            values = {"Q": 100.0, "R": 1.0, "C": 1.0, "B": 0.0, "V": 1.0}

        # Run optimizer
        # mapping bins to correct format
        # bins argument in run_vrpp_optimizer expects NDArray[float] of amounts

        route, profit = run_vrpp_optimizer(
            bins=bins,
            distance_matrix=distance_matrix,
            param=kwargs.get("param", 0.0),
            media=kwargs.get("media", np.array([])),
            desviopadrao=kwargs.get("desviopadrao", np.array([])),
            values=values,
            binsids=binsids,
            must_go=must_go,
            number_vehicles=kwargs.get("number_vehicles", 1),
            time_limit=kwargs.get("time_limit", 60),
            optimizer=kwargs.get("optimizer", "gurobi"),
        )

        return route, profit
