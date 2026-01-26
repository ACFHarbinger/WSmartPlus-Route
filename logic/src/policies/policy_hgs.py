"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.hybrid_genetic_search import run_hgs


@PolicyRegistry.register("policy_hgs")
class HGSPolicy(IPolicy):
    """
    Hybrid Genetic Search policy class.
    Executes HGS for VRP.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the HGS policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        # kwargs["waste_type"] and kwargs["area"] are available but not directly used
        # unless passed down via config if needed.
        config = kwargs.get("config", {})

        # 1. Determine Must-Go Bins (VRPP Logic)
        try:
            # Pattern: policy_hgs_<threshold>
            threshold_std = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            threshold_std = 1.0  # Default

        if not hasattr(bins, "means") or bins.means is None:
            raise ValueError("Bins object missing 'means' attribute.")
        else:
            means = bins.means
            std = bins.std

        current_fill = bins.c
        predicted_fill = current_fill + means + (threshold_std * std)

        # Must-go bins
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        # 2. Prepare Data for HGS
        # HGS visits all nodes in the passed matrix/demands.
        # We must subset for just the target bins.

        target_nodes = must_go_indices
        if not target_nodes:
            return [0, 0], 0.0, None

        # Build demands dict {node_idx: demand}
        # Note: demands dict keys will be 1..K (new indices)
        demands = {i + 1: current_fill[i] for i in range(len(current_fill))}  # Full demands

        # HGS Config
        hgs_config = config.get("hgs", {})

        # Engine override
        if "engine" in kwargs:
            hgs_config["engine"] = kwargs["engine"]

        capacity = hgs_config.get("capacity", 100.0)
        revenue = hgs_config.get("revenue", 1.0)
        cost_unit = hgs_config.get("cost_unit", 1.0)

        # Subset mapping
        real_target_indices = [idx + 1 for idx in target_nodes]
        subset_indices = [0] + real_target_indices

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        sub_demands = {i: demands[original_idx] for i, original_idx in enumerate(real_target_indices, 1)}

        # Run HGS
        best_routes, _, _ = run_hgs(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, hgs_config)

        # Map routes back
        tour = [0]
        if best_routes:
            for route in best_routes:
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
