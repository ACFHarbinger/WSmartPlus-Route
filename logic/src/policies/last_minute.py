"""
Last-Minute (Reactive) routing policy module.

This module implements reactive waste collection policies that trigger collection
only when bin fill levels exceed a specified threshold. This is a demand-driven
approach that responds to actual waste accumulation rather than following a fixed schedule.

The module provides two variants:
1. policy_last_minute: Collect bins above threshold
2. policy_last_minute_and_path: Collect threshold bins + opportunistic collection
   along the shortest paths between visited bins (reduces future trips)

Key Features:
- Threshold-based triggering (e.g., collect when fill level > 80%)
- Single and multi-vehicle support
- Capacity-aware routing (insert depot returns when vehicle full)
- Opportunistic collection along paths for efficiency

This policy is useful for:
- Sparse or irregular waste generation patterns
- Reducing unnecessary trips to nearly-empty bins
- Benchmarking against fixed-schedule policies
"""
import numpy as np

from typing import List
from pandas import DataFrame
from numpy.typing import NDArray
from .multi_vehicle import find_routes
from .single_vehicle import find_route, get_multi_tour
from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params


def policy_last_minute(
        bins: NDArray[np.float64],
        distancesC: NDArray[np.int32],
        lvl: NDArray[np.float64],
        waste_type: str='plastic',
        area: str='riomaior',
        n_vehicles: int=1,
        coords: DataFrame=None
    ):
    """
    Execute a last-minute (reactive) collection policy.

    Collects only bins whose fill level exceeds the threshold. If any bins
    exceed the threshold, generates a route covering all such bins.

    Args:
        bins (NDArray[np.float64]): Current fill levels for all bins (0-100%)
        distancesC (NDArray[np.int32]): Distance matrix for routing
        lvl (NDArray[np.float64]): Threshold fill levels per bin (same shape as bins)
        waste_type (str): Type of waste being collected. Default: 'plastic'
        area (str): Geographic area name. Default: 'riomaior'
        n_vehicles (int): Number of vehicles available. Default: 1
        coords (DataFrame, optional): Bin coordinates (for multi-vehicle routing)

    Returns:
        List[int]: Tour as a sequence of node IDs. Returns [0] if no bins exceed threshold.
    """
    tour = []
    to_collect = np.nonzero(bins > lvl)[0] + 1
    if len(to_collect) > 0:
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = find_route(distancesC, to_collect)
            tour = get_multi_tour(tour, bins, max_capacity, distancesC)
        else:
           tour = find_routes(distancesC, bins, max_capacity, to_collect, n_vehicles, coords)
    else:
        tour = [0]
    return tour


def policy_last_minute_and_path(
        bins: NDArray[np.float64],
        distancesC: NDArray[np.int32],
        paths_between_states: List[List[int]],
        lvl: NDArray[np.float64],
        waste_type: str='plastic',
        area: str='riomaior',
        n_vehicles: int=1,
        coords: DataFrame=None
    ):
    """
    Execute last-minute policy with opportunistic path-based collection.

    Similar to policy_last_minute, but also opportunistically collects bins
    along the shortest paths between threshold-exceeding bins, if capacity allows.
    This reduces future collection trips by serving nearby bins "on the way".

    Algorithm:
    1. Identify bins exceeding threshold
    2. Generate initial route covering these bins
    3. For each edge in the route, check bins along the shortest path
    4. Collect path bins if they haven't been visited and capacity allows

    Args:
        bins (NDArray[np.float64]): Current fill levels for all bins (0-100%)
        distancesC (NDArray[np.int32]): Distance matrix for routing
        paths_between_states (List[List[int]]): Precomputed shortest paths.
            paths_between_states[i][j] is the path from node i to node j
        lvl (NDArray[np.float64]): Threshold fill levels per bin
        waste_type (str): Type of waste being collected. Default: 'plastic'
        area (str): Geographic area name. Default: 'riomaior'
        n_vehicles (int): Number of vehicles available. Default: 1
        coords (DataFrame, optional): Bin coordinates (for multi-vehicle routing)

    Returns:
        List[int]: Tour including threshold bins and opportunistic path bins.
            Returns [0] if no bins exceed threshold.
    """
    tour = []
    to_collect = np.nonzero(bins > lvl)[0] + 1
    if len(to_collect) > 0:
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = find_route(distancesC, to_collect)
        else:
           tour = find_routes(distancesC, bins, max_capacity, to_collect, n_vehicles, coords)
        visited_states = [0]
        len_tour = len(tour)
        np_tour = np.array(tour)
        total_waste = np.sum(bins[np_tour[np_tour != 0] - 1])
        for ii in range(0, len_tour - 1):
            path_to_collect = paths_between_states[tour[ii]][tour[ii+1]]
            for tocol in path_to_collect:
                if tocol not in tour and tocol != 0 and tocol not in visited_states:
                    waste = bins[tocol - 1]
                    if waste + total_waste <= max_capacity:
                        total_waste += waste
                        visited_states.append(tocol)
                elif tocol not in visited_states:
                    visited_states.append(tocol)
        
        #del visited_states[0]
        if len(visited_states) > 1: visited_states.append(0)
        tour = get_multi_tour(visited_states, bins, max_capacity, distancesC)
    else:
        tour = [0]
    return tour


def policy_profit_reactive(
        bins: NDArray[np.float64],
        distancesC: NDArray[np.int32],
        waste_type: str = 'plastic',
        area: str = 'riomaior',
        n_vehicles: int = 1,
        coords: DataFrame = None,
        profit_threshold: float = 0.0
    ):
    """
    Execute a profit-based reactive collection policy.

    Collects bins only if their individual expected reward (waste * revenue_kg) 
    exceeds a certain profit threshold.
    """
    (vehicle_capacity, revenue_kg, density, cost_km, volume) = load_area_and_waste_type_params(area, waste_type)
    bin_capacity = volume * density
    
    # Calculate expected revenue per bin (in currency units)
    # bins is 0-100%, so we multiply by bin_capacity / 100
    expected_revenue = (bins / 100.0) * bin_capacity * revenue_kg
    
    to_collect = np.nonzero(expected_revenue > profit_threshold)[0] + 1
    
    tour = []
    if len(to_collect) > 0:
        if n_vehicles == 1:
            tour = find_route(distancesC, to_collect)
            tour = get_multi_tour(tour, bins, vehicle_capacity, distancesC)
        else:
           tour = find_routes(distancesC, bins, vehicle_capacity, to_collect, n_vehicles, coords)
    else:
        tour = [0]
    return tour
