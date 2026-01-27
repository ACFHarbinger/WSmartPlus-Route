"""
LKH Policy Adapter.
"""
from typing import Any, List, Tuple

import numpy as np

from .adapters import IPolicy, PolicyRegistry
from .lin_kernighan import solve_lk
from .single_vehicle import get_route_cost


@PolicyRegistry.register("lkh")
class LKHPolicy(IPolicy):
    """
    Lin-Kernighan heuristic policy class.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LKH policy.
        """
        must_go = kwargs.get("must_go", [])
        if not must_go:
            return [0, 0], 0.0, None

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        config = kwargs.get("config", {})
        lkh_config = config.get("lkh", {})

        # 1. Map to local subset
        map_local_to_global = {0: 0}
        for idx, i in enumerate(must_go):
            map_local_to_global[idx + 1] = i + 1

        n_nodes = len(must_go) + 1
        sub_matrix = np.zeros((n_nodes, n_nodes))
        local_waste = np.zeros(n_nodes)

        # Load params (need B, E for specific weight calc if needed, or just % fill)
        # Using bins.c for simplicity or loading if required.
        for r in range(n_nodes):
            orig_node_idx = map_local_to_global[r]
            local_waste[r] = bins.c[orig_node_idx - 1] if r > 0 else 0
            for c in range(n_nodes):
                sub_matrix[r, c] = distance_matrix[map_local_to_global[r]][map_local_to_global[c]]

        capacity = lkh_config.get("capacity", 100.0)
        lk_tour_local, _ = solve_lk(sub_matrix, waste=local_waste, capacity=capacity)
        lk_tour_global = [map_local_to_global[i] for i in lk_tour_local]

        # Final capacity check
        if lkh_config.get("check_capacity", True):
            from .single_vehicle import get_multi_tour

            lk_tour_global = get_multi_tour(lk_tour_global, bins.c, capacity, distance_matrix)

        return lk_tour_global, get_route_cost(distance_matrix, lk_tour_global), None
