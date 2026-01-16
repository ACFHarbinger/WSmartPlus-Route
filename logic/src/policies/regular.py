"""
Regular (Periodic) routing policy module.

This module implements a periodic waste collection policy where all bins in the
service area are visited on a fixed schedule (e.g., every N days). This is the
traditional approach used in many municipal waste collection systems.

The policy:
1. Visits all bins on specific days (based on the level parameter)
2. Generates a route covering all collection points
3. Supports both single and multi-vehicle routing
4. Can cache routes for repeated use (route stability)

This is often used as a baseline policy for comparison against more
sophisticated approaches like neural models or optimization-based policies.
"""

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params

from .multi_vehicle import find_routes
from .single_vehicle import find_route, get_multi_tour


def policy_regular(
    n_bins: int,
    bins_waste: NDArray[np.float64],
    distancesC: NDArray[np.int32],
    lvl: int,
    day: int,
    cached: Optional[List[int]] = [],
    waste_type: str = "plastic",
    area: str = "riomaior",
    n_vehicles: int = 1,
    coords: DataFrame = None,
):
    """
    Execute a regular periodic collection policy.

    Visits all bins on a fixed schedule: every (lvl + 1) days starting from day 1.
    For example, lvl=6 means collection every 7 days (weekly).

    The policy generates a TSP/VRP tour covering all bins on collection days.
    Routes can be cached for efficiency (same route used repeatedly).

    Args:
        n_bins (int): Number of bins in the service area
        bins_waste (NDArray[np.float64]): Current waste levels in bins
        distancesC (NDArray[np.int32]): Distance matrix for routing
        lvl (int): Collection frequency level (collection every lvl+1 days)
        day (int): Current simulation day
        cached (Optional[List[int]]): Cached route from previous collection day.
            If provided and non-empty, reuses this route. Default: []
        waste_type (str): Type of waste (e.g., 'plastic', 'organic'). Default: 'plastic'
        area (str): Geographic area name (e.g., 'riomaior'). Default: 'riomaior'
        n_vehicles (int): Number of vehicles available. Default: 1
        coords (DataFrame, optional): Bin coordinates (for multi-vehicle routing)

    Returns:
        List[int]: Tour as a sequence of node IDs. Returns [0] if not a collection day.
    """
    tour = []
    if (day % (lvl + 1)) == 1:
        to_collect = np.arange(0, n_bins, dtype="int32") + 1
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = cached if cached is not None and len(cached) > 1 else find_route(distancesC, to_collect)
            tour = get_multi_tour(tour, bins_waste, max_capacity, distancesC)
        else:
            tour = (
                cached
                if cached is not None and len(cached) > 1
                else find_routes(distancesC, bins_waste, max_capacity, to_collect, n_vehicles, coords)
            )
    else:
        tour = [0]
    return tour
