"""
CVRP Policy module.

Implements a multi-vehicle routing policy (CVRP) that visits all bins.
Use this policy when n_vehicles > 1 or default multi-vehicle behavior is desired.
"""

from typing import Any, List, Tuple

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.multi_vehicle import find_routes
from logic.src.policies.single_vehicle import get_route_cost


@PolicyRegistry.register("cvrp")
class CVRPPolicy(IPolicy):
    """
    Capacitated Vehicle Routing Policy (CVRP).
    Visits all bins using multiple vehicles.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the CVRP policy.
        """
        # Unpack arguments
        bins = kwargs["bins"]
        distancesC = kwargs["distancesC"]
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        n_vehicles = kwargs.get("n_vehicles", 1)  # Default to passed value (usually from env or config)
        cached = kwargs.get("cached")
        coords = kwargs.get("coords")  # Needed for multi-vehicle heuristic?

        # Determine capacity
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)

        bins_waste = bins.c
        n_bins = len(bins_waste)

        # All bins 1..N
        to_collect = list(range(1, n_bins + 1))

        # Run CVRP Logic
        # Explicit (n_vehicles >= 1) logic, but typically used for > 1
        # find_routes handles partitioning.

        if cached is not None and len(cached) > 1:
            tour = cached
        else:
            # Note: find_routes signature: (distances, bin_waste, capacity, tour_indices, n_vehicles, coords)
            tour = find_routes(distancesC, bins_waste, max_capacity, to_collect, n_vehicles, coords)

        # Calculate cost
        cost = get_route_cost(distancesC, tour)

        return tour, cost, tour
