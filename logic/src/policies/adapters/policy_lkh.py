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

        # Load area parameters (capacity, revenue, etc.)
        capacity, _, _, _ = self._load_area_params(area, waste_type, config)

        # Handle nested configuration
        lkh_cfg = config.get("lkh", {})
        engine = "custom"

        values = {"check_capacity": True, "max_iterations": 100}
        if engine in lkh_cfg:
            opt_cfg = lkh_cfg[engine]
            if isinstance(opt_cfg, list):
                for item in opt_cfg:
                    if isinstance(item, dict):
                        values.update(item)
            elif isinstance(opt_cfg, dict):
                values.update(opt_cfg)
        else:
            values.update(lkh_cfg)

        max_iterations = values.get("max_iterations", 100)
        check_capacity = values.get("check_capacity", True)

        # Map to local subset
        # subset_indices[0] = depot (0), subset_indices[1..M] = must_go bins
        map_local_to_global = {0: 0}
        for idx, i in enumerate(must_go):
            # i is already a 1-based bin ID
            map_local_to_global[idx + 1] = i

        n_nodes = len(must_go) + 1
        sub_matrix = np.zeros((n_nodes, n_nodes))
        local_waste = np.zeros(n_nodes)

        for r in range(n_nodes):
            orig_node_idx = map_local_to_global[r]
            # orig_node_idx is 1-based ID for bins, 0 for depot
            local_waste[r] = bins.c[orig_node_idx - 1] if r > 0 else 0
            for c in range(n_nodes):
                sub_matrix[r, c] = distance_matrix[map_local_to_global[r]][map_local_to_global[c]]

        lk_tour_local, _ = solve_lk(
            sub_matrix,
            waste=local_waste,
            capacity=capacity,
            max_iterations=max_iterations,
        )
        lk_tour_global = [map_local_to_global[i] for i in lk_tour_local]

        # Final capacity check/splitting
        if check_capacity:
            lk_tour_global = get_multi_tour(lk_tour_global, bins.c, capacity, distance_matrix)

        cost = get_route_cost(distance_matrix, lk_tour_global)
        return lk_tour_global, cost, None
