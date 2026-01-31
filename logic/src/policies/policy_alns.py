"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""
from typing import Any, List, Tuple

import numpy as np

from .adapters import IPolicy, PolicyRegistry
from .adaptive_large_neighborhood_search import run_alns, run_alns_ortools, run_alns_package
from .single_vehicle import get_route_cost


@PolicyRegistry.register("alns")
class ALNSPolicy(IPolicy):
    """
    ALNS policy class.
    Visits pre-selected 'must_go' bins.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the ALNS policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        policy_name = kwargs.get("policy", "alns")
        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        # Load Area Parameters
        from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        Q, R, _, C, _ = load_area_and_waste_type_params(area, waste_type)

        alns_config = config.get("alns", {})
        capacity = alns_config.get("capacity", Q)
        revenue = alns_config.get("revenue", R)
        cost_unit = alns_config.get("cost_unit", C)

        values = {"Q": capacity, "R": revenue, "C": cost_unit}
        values.update(alns_config)

        # Prepare Data for ALNS
        # must_go contains 1-based IDs. We visit ALL these bins.
        subset_indices = [0] + list(must_go)
        dist_matrix_np = np.array(distance_matrix)
        sub_dist_matrix = dist_matrix_np[np.ix_(subset_indices, subset_indices)]

        # Build local demands. ALNS backend (custom/ortools) visits everyone in demands.
        sub_demands = {}
        for i, global_idx in enumerate(must_go, 1):
            fill = bins.c[global_idx - 1]
            sub_demands[i] = float(fill)

        # 1. Dispatch variant
        if "package" in policy_name:
            routes, _, _ = run_alns_package(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, values)
        elif "ortools" in policy_name:
            routes, _, _ = run_alns_ortools(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, values)
        else:
            routes, _, _ = run_alns(sub_dist_matrix, sub_demands, capacity, revenue, cost_unit, values)

        # 2. Map back to global IDs
        tour = [0]
        if routes:
            for r in routes:
                for node_idx in r:
                    tour.append(subset_indices[node_idx])
                tour.append(0)

        if len(tour) == 1:
            tour = [0, 0]

        return tour, get_route_cost(distance_matrix, tour), None
