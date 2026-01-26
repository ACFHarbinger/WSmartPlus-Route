"""
BCP Policy Adapter.

Adapts the Branch-Cut-and-Price (BCP) logic to the common policy interface.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.branch_cut_and_price import run_bcp


@PolicyRegistry.register("policy_bcp")
class BCPPolicy(IPolicy):
    """
    BCP policy class.
    Executes Branch-Cut-and-Price logic.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the BCP policy.
        """
        # policy is like "policy_bcp_<param>"
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})

        current_fill = bins.c
        if hasattr(bins, "means") and bins.means is not None:
            means = bins.means
            std = bins.std
        else:
            means = np.zeros_like(current_fill)
            std = np.zeros_like(current_fill)  # Should handle error if needed

        try:
            param = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            param = 1.0

        predicted_fill = current_fill + means + (param * std)
        # Must-go logical: if prediction > 100 or current > 100
        # Indices in 0..N-1 range
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        # The run_bcp function expects:
        # matrix, demands (dict?), capacity, revenue, cost_unit, values?
        # Let's check signature from previous read:
        # run_bcp(distance_matrix, demands, capacity, R, C, values, must_go_indices)

        bcp_config = config.get("bcp", {})

        # Helper to override engine
        if "engine" in kwargs:
            bcp_config["bcp_engine"] = kwargs["engine"]

        capacity = bcp_config.get("capacity", 100.0)
        revenue = bcp_config.get("revenue", 1.0)
        cost_unit = bcp_config.get("cost_unit", 1.0)

        # Demands dictionary {1: val, 2: val...} matching 1..N indices
        demands = {i + 1: current_fill[i] for i in range(len(current_fill))}

        best_routes, cost = run_bcp(distance_matrix, demands, capacity, revenue, cost_unit, bcp_config, must_go_indices)

        # run_bcp returns routes as List[List[int]], cost
        # Flatten to single tour [0, ... 0, ... 0]
        tour = [0]
        if best_routes:
            for r in best_routes:
                # r contains node indices
                tour.extend(r)
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour, cost, None
