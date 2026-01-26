"""
TSP Policy module.

Implements a single-vehicle routing policy (TSP) that visits all bins.
Use this policy when n_vehicles=1 is enforced or desired.
"""

from typing import Any, List, Tuple

from logic.src.pipeline.simulations.loader import load_area_and_waste_type_params
from logic.src.policies.adapters import IPolicy, PolicyRegistry
from logic.src.policies.single_vehicle import find_route, get_multi_tour


@PolicyRegistry.register("tsp")
class TSPPolicy(IPolicy):
    """
    Traveling Salesperson Policy (TSP).
    Visits all bins using a single vehicle strategy.
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the TSP policy.
        """
        # Unpack arguments
        bins = kwargs["bins"]
        distancesC = kwargs["distancesC"]
        waste_type = kwargs["waste_type"]
        area = kwargs["area"]
        cached = kwargs.get("cached")

        # Determine capacity
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)

        bins_waste = bins.c
        n_bins = len(bins_waste)

        # All bins 1..N (depot is 0)
        # Note: distancesC is (N+1)x(N+1). Bins are indices 1 to n_bins.
        to_collect = list(range(1, n_bins + 1))

        # Run TSP Logic (n_vehicles == 1)
        if cached is not None and len(cached) > 1:
            tour = cached
        else:
            tour = find_route(distancesC, to_collect)

        # Ensure capacity constraints -> separate into multiple trips if needed
        # Single vehicle doing multiple rounds?
        tour = get_multi_tour(tour, bins_waste, max_capacity, distancesC)

        # Calculate cost
        # Logic matches `regular.py`: cost calculated outside or by `get_route_cost`?
        # execute returns (tour, cost, extra/cached)
        # We need to calculate cost of the final tour.
        # Since `tour` might be [0, 1, 2, 0, 3, 4, 0] (multi-trip), we sum segments.

        # Import local utility for cost calculation?
        # Or just do it manually. `distancesC` is ints.
        # `single_vehicle.get_route_cost` takes `distances` and `tour`.
        from logic.src.policies.single_vehicle import get_route_cost

        cost = get_route_cost(distancesC, tour)

        return tour, cost, tour  # Return tour as 'cached' for next step? Regular policy does.
