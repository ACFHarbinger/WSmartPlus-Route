"""
TSP Policy module.

Implements a single-vehicle routing policy (TSP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ..base_routing_policy import BaseRoutingPolicy
from ..single_vehicle import find_route, get_multi_tour
from .factory import PolicyRegistry


@PolicyRegistry.register("tsp")
class TSPPolicy(BaseRoutingPolicy):
    """
    Traveling Salesperson Policy (TSP).

    Visits provided 'must_go' bins using a single vehicle strategy with
    capacity-based tour splitting.
    """

    def _get_config_key(self) -> str:
        """Return config key for TSP."""
        return "tsp"

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
        """
        Run TSP solver with capacity-based tour splitting.

        Note: TSP uses 1-based node IDs directly (not the local subset mapping),
        so we need to handle the indices differently.
        """
        # Get the subset_indices from context for mapping
        must_go = kwargs.get("must_go", [])
        distancesC = kwargs.get("distancesC")
        bins = kwargs.get("bins")

        # Use distancesC if available (TSP uses integer distances)
        dist_matrix = distancesC if distancesC is not None else kwargs.get("distance_matrix")

        # TSP works directly with must_go indices (1-based global IDs)
        to_collect = list(must_go) if must_go else []

        if not to_collect:
            return [[]], 0.0

        # Find TSP route
        tour = find_route(dist_matrix, to_collect)

        # Ensure bins is not None
        if bins is None:
            return [[]], 0.0

        # Split by capacity
        tour = get_multi_tour(tour, bins.c, capacity, dist_matrix)

        # Convert tour to route format expected by base class
        # Tour is [0, a, b, ..., 0] format, we return as list of routes
        routes: List[List[int]] = []
        if tour and len(tour) > 2:
            current_route: List[int] = []
            for node in tour:
                if node == 0:
                    if current_route:
                        routes.append(current_route)
                        current_route = []
                else:
                    current_route.append(node)
            if current_route:
                routes.append(current_route)

        return routes, 0.0

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute TSP policy.

        Overrides base execute because TSP has unique handling for cached tours
        and uses different distance matrix key.
        """
        must_go = kwargs.get("must_go", [])
        early_result = self._validate_must_go(must_go)
        if early_result is not None:
            return early_result

        bins = kwargs["bins"]
        area = kwargs.get("area", "Rio Maior")
        waste_type = kwargs.get("waste_type", "plastic")
        cached = kwargs.get("cached")
        config = kwargs.get("config", {})
        distancesC = kwargs.get("distancesC")
        distance_matrix = kwargs.get("distance_matrix", distancesC)

        # Load capacity
        capacity, _, _, values = self._load_area_params(area, waste_type, config)

        # Use cached route if available and no specific must_go
        if cached is not None and len(cached) > 1 and not must_go:
            tour = cached
        else:
            to_collect = list(must_go) if must_go else list(range(1, bins.n + 1))
            time_limit = values.get("time_limit", 2.0)
            tour = find_route(distancesC, to_collect, time_limit=time_limit)
            tour = get_multi_tour(tour, bins.c, capacity, distancesC)

        # Ensure list format
        if hasattr(tour, "tolist"):
            tour = tour.tolist()
        elif not isinstance(tour, list):
            tour = list(tour)

        cost = self._compute_cost(distance_matrix, tour)
        return tour, cost, tour
