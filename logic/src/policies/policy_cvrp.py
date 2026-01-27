"""
CVRP Policy module.

Implements a multi-vehicle routing policy (CVRP) that visits a specific set of bins.
Agnostic to how the targets were selected.
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params

from .adapters import IPolicy, PolicyRegistry
from .multi_vehicle import find_routes
from .single_vehicle import get_route_cost


@PolicyRegistry.register("cvrp")
class CVRPPolicy(IPolicy):
    """
    Capacitated Vehicle Routing Policy (CVRP).
    Visits provided 'must_go' bins using multiple vehicles.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the CVRP policy.
        """
        bins = kwargs["bins"]
        distancesC = kwargs["distancesC"]
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        n_vehicles = kwargs.get("n_vehicles", 1)
        cached = kwargs.get("cached")
        coords = kwargs.get("coords")
        must_go = kwargs.get("must_go", [])

        to_collect = must_go if must_go else list(range(1, bins.n + 1))

        if not to_collect:
            return [0, 0], 0.0, None

        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)

        if cached is not None and len(cached) > 1 and not must_go:
            tour = cached
        else:
            tour = find_routes(distancesC, bins.c, max_capacity, np.array(to_collect), n_vehicles, coords)

        distance_matrix = kwargs.get("distance_matrix", distancesC)
        cost = get_route_cost(distance_matrix, tour)

        return tour, cost, tour
