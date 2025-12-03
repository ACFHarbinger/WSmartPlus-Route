import numpy as np

from numpy.typing import NDArray
from typing import Optional, List
from .multi_vehicle import find_routes
from .single_vehicle import find_route, get_multi_tour
from src.pipeline.simulator.loader import load_area_and_waste_type_params


def policy_regular(
        n_bins: int,
        bins_waste: NDArray[np.float64], 
        distancesC: NDArray[np.int32], 
        lvl: int, 
        day: int, 
        cached: Optional[List[int]]=[],
        waste_type: str='plastic', 
        area: str='riomaior', 
        n_vehicles: int=1,
        *args
    ):
    tour = []
    if (day % (lvl + 1)) == 1:
        to_collect = np.arange(0, n_bins, dtype="int32") + 1
        max_capacity, _, _, _, _ = load_area_and_waste_type_params(area, waste_type)
        if n_vehicles == 1:
            tour = cached if cached is not None and len(cached) > 1 else find_route(distancesC, to_collect)
            tour = get_multi_tour(tour, bins_waste, max_capacity, distancesC)
        else:
           demands, depot = args
           tour, cost = find_routes(distancesC, demands, max_capacity, to_collect, n_vehicles, depot)
    else:
        tour = [0]
    return tour
