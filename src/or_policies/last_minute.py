import numpy as np

from typing import List
from numpy.typing import NDArray
from .multi_vehicle import find_routes
from .single_vehicle import find_route, get_multi_tour
from src.pipeline.simulator.loader import load_area_and_waste_type_params


def policy_last_minute(
        bins: NDArray[np.float64], 
        distancesC: NDArray[np.int32], 
        lvl: NDArray[np.float64],
        waste_type: str='plastic', 
        area: str='riomaior', 
        n_vehicles: int=1,
        *args
    ):
    tour = []
    to_collect = np.nonzero(bins > lvl)[0]
    if len(to_collect) > 0:
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = find_route(distancesC, to_collect + 1)
            tour = get_multi_tour(tour, bins, max_capacity, distancesC)
        else:
           demands, depot = args
           tour, cost = find_routes(distancesC, demands, max_capacity, to_collect, n_vehicles, depot)
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
        *args
    ):
    tour = []
    to_collect = np.nonzero(bins > lvl)[0]
    if len(to_collect) > 0:
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = find_route(distancesC, to_collect + 1)
        else:
           demands, depot = args
           tour, cost = find_routes(distancesC, demands, max_capacity, to_collect, n_vehicles, depot)
        visited_states = [0]
        len_tour = len(tour)
        np_tour = np.array(tour)
        total_waste = np.sum(bins[np_tour[np_tour != 0] - 1])
        for ii in range(0, len_tour - 1):
            path_to_collect = paths_between_states[tour[ii]][tour[ii+1]]
            for tocol in path_to_collect:
                if tocol not in tour and tocol != 0:
                    waste = bins[tocol]
                    if waste + total_waste <= max_capacity:
                        total_waste += waste
                        #print("Waste collected trip {}: {}".format(tocol, total_waste))
                        visited_states.append(tocol)
                elif tocol not in visited_states:
                    visited_states.append(tocol)
        
        #del visited_states[0]
        if len(visited_states) > 1: visited_states.append(0)
        tour = get_multi_tour(visited_states, bins, max_capacity, distancesC)
    else:
        tour = [0]
    return tour
