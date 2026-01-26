"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the common policy interface.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.adaptive_large_neighborhood_search import run_alns


@PolicyRegistry.register("policy_alns")
class ALNSPolicy(IPolicy):
    """
    ALNS policy class.
    Executes Adaptive Large Neighborhood Search for VRP.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the ALNS policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_alns_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0  # Default to 1.0 std dev if not specified

        # Load parameters for prediction if not present in bins object
        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means'/ 'std' attributes required for prediction.")
        else:
            means = bins.means
            std = bins.std

        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        # Must-go bins: predicted >= 100%
        # Also include currently overflowing bins
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        # 2. Prepare Data for ALNS

        demands = {i + 1: current_fill[i] for i in range(len(current_fill))}

        # ALNS Config
        alns_config = config.get("alns", {})

        # Engine override
        if "engine" in kwargs:
            alns_config["engine"] = kwargs["engine"]

        capacity = alns_config.get("capacity", 100.0)  # Vehicle capacity
        revenue = alns_config.get("revenue", 1.0)
        cost_unit = alns_config.get("cost_unit", 1.0)

        # Subset mapping for Must-Go
        target_nodes = must_go_indices
        if not target_nodes:
            return [0, 0], 0.0, None

        real_target_indices = [idx + 1 for idx in target_nodes]
        subset_indices = [0] + real_target_indices

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        sub_demands = {i: demands[original_idx] for i, original_idx in enumerate(real_target_indices, 1)}

        # Run ALNS
        # run_alns signature: (dist_matrix, demands, capacity, R, C, values)
        best_routes, _, _ = run_alns(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, alns_config)

        # Map routes back
        tour = [0]
        if best_routes:
            for route_idx, route in enumerate(best_routes):
                for node_idx in route:
                    original_matrix_idx = subset_indices[node_idx]
                    tour.append(original_matrix_idx)
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        # Recalculate cost
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i + 1]]

        return tour, cost, None
