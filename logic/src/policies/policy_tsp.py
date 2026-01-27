"""
TSP Policy module.

Implements a single-vehicle routing policy (TSP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

from .adapters import IPolicy, PolicyRegistry
from .single_vehicle import find_route, get_multi_tour, get_route_cost


@PolicyRegistry.register("tsp")
class TSPPolicy(IPolicy):
    """
    Traveling Salesperson Policy (TSP).
    Visits provide 'must_go' bins using a single vehicle strategy.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the TSP policy.
        """
        bins = kwargs["bins"]
        distancesC = kwargs["distancesC"]
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        cached = kwargs.get("cached")
        must_go = kwargs.get("must_go", [])

        # If no must_go provided, fallback to all bins (legacy behavior)
        # but in modular mode, this typically won't be empty.
        to_collect = must_go if must_go else list(range(1, bins.n + 1))

        if not to_collect:
            return [0, 0], 0.0, None

        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)

        if cached is not None and len(cached) > 1 and not must_go:
            tour = cached
        else:
            tour = find_route(distancesC, np.array(to_collect))

        # Handle capacity
        tour = get_multi_tour(tour, bins.c, max_capacity, distancesC)

        distance_matrix = kwargs.get("distance_matrix", distancesC)
        cost = get_route_cost(distance_matrix, tour)

        return tour, cost, tour
