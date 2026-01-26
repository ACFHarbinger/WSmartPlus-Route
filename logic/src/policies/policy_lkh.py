"""
LKH Policy Adapter.

Adapts the Lin-Kernighan Heuristic (LKH) logic to the common policy interface.
"""
from typing import Any, List, Tuple

import numpy as np

from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.lin_kernighan import solve_lk


@PolicyRegistry.register("lkh")
class LKHPolicy(IPolicy):
    """
    Lin-Kernighan Heuristic policy class.
    Executes LKH for VRP/TSP.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LKH policy.
        """
        policy = kwargs["policy"]
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})

        # Standard VRPP logic for subset selection
        try:
            param = float(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            param = 1.0

        if hasattr(bins, "means") and bins.means is not None:
            means = bins.means
            std = bins.std
        else:
            means = np.zeros_like(bins.c)
            std = np.zeros_like(bins.c)

        current_fill = bins.c
        predicted_fill = current_fill + means + (param * std)
        must_go_indices = np.where((predicted_fill >= 100.0) | (current_fill >= 100.0))[0].tolist()

        target_nodes = must_go_indices
        if not target_nodes:
            return [0, 0], 0.0, None

        # LKH Config
        lkh_config = config.get("lkh", {}).copy()
        capacity = lkh_config.get("capacity", 100.0)

        # Subset mapping
        real_target_indices = [idx + 1 for idx in target_nodes]
        subset_indices = [0] + real_target_indices

        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Prepare Waste array for sub-problem
        sub_waste = [0.0]  # Depot
        for original_idx in real_target_indices:
            sub_waste.append(current_fill[original_idx - 1])

        lkh_config["waste"] = np.array(sub_waste)
        lkh_config["capacity"] = capacity

        # Run LKH
        best_tour, _ = solve_lk(sub_dist_matrix, lkh_config)

        # Map tour back
        tour = []
        if best_tour:
            for node_idx in best_tour:
                original_matrix_idx = subset_indices[node_idx]
                tour.append(original_matrix_idx)

        # Ensure valid structure
        if not tour:
            tour = [0, 0]
        elif tour[0] != 0:
            tour = [0] + tour

        if tour[-1] != 0:
            tour.append(0)

        if len(tour) <= 2:
            # If just [0, 0] or [0]
            tour = [0, 0]

        # Recalculate cost
        cost = 0.0
        for i in range(len(tour) - 1):
            cost += distance_matrix[tour[i]][tour[i + 1]]

        return tour, cost, None
