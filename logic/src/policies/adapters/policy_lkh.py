"""
LKH Policy Adapter.

Uses Lin-Kernighan heuristic for TSP optimization.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..base_routing_policy import BaseRoutingPolicy
from ..lin_kernighan import solve_lk
from ..single_vehicle import get_multi_tour, get_route_cost
from .factory import PolicyRegistry


@PolicyRegistry.register("lkh")
class LKHPolicy(BaseRoutingPolicy):
    """
    Lin-Kernighan heuristic policy class.

    Uses LK-tour improvement for TSP with capacity-based splitting.
    """

    def _get_config_key(self) -> str:
        """Return config key for LKH."""
        return "lkh"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_demands: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float]:
        """Not used - LKH requires specialized execute()."""
        return [[]], 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the LKH policy.

        Uses specialized local-to-global mapping for LK solver.
        """
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        distance_matrix = kwargs["distance_matrix"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        config = kwargs.get("config", {})
        lkh_config = config.get("lkh", {})

        # Load capacity
        capacity, _, _, _ = self._load_area_params(area, waste_type, config)

        # Map to local subset
        map_local_to_global = {0: 0}
        for idx, i in enumerate(must_go):
            map_local_to_global[idx + 1] = i + 1

        n_nodes = len(must_go) + 1
        sub_matrix = np.zeros((n_nodes, n_nodes))
        local_waste = np.zeros(n_nodes)

        for r in range(n_nodes):
            orig_node_idx = map_local_to_global[r]
            local_waste[r] = bins.c[orig_node_idx - 1] if r > 0 else 0
            for c in range(n_nodes):
                sub_matrix[r, c] = distance_matrix[map_local_to_global[r]][map_local_to_global[c]]

        lk_tour_local, _ = solve_lk(sub_matrix, waste=local_waste, capacity=capacity)
        lk_tour_global = [map_local_to_global[i] for i in lk_tour_local]

        # Final capacity check
        if lkh_config.get("check_capacity", True):
            lk_tour_global = get_multi_tour(lk_tour_global, bins.c, capacity, distance_matrix)

        cost = get_route_cost(distance_matrix, lk_tour_global)
        return lk_tour_global, cost, None
