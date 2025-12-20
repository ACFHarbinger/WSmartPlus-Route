import torch
import numpy as np
import scipy.stats as stats

from logic.src.utils.definitions import MAX_WASTE, MAX_LENGTHS
from logic.src.utils.functions import get_path_until_string
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.processor import process_coordinates


__all__ = [
    'generate_vrpp_data',
    'generate_wcvrp_data',
]


def generate_vrpp_data(dataset_size, vrpp_size, waste_type, 
                       distribution="gamma1", area="Rio Maior", focus_graph=None, focus_size=0, method=None, num_days=1):
    if focus_graph is not None:
        assert focus_size > 0
        depot, loc, mm_arr, idx = load_focus_coords(vrpp_size, method, area, waste_type, focus_graph, focus_size)
        remaining_coords_size = dataset_size - focus_size
        random_coords = np.random.uniform(mm_arr[0], mm_arr[1], size=(remaining_coords_size, vrpp_size+1, mm_arr.shape[-1]))
        depots, locs = process_coordinates(random_coords, method, col_names=None)
        depot = np.concatenate((depot, depots))
        loc = np.concatenate((loc, locs))
        assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]
        if distribution == 'emp':
            data_dir = get_path_until_string(focus_graph, 'wsr_simulator')
            bins = Bins(vrpp_size, data_dir, sample_dist=distribution, area=area, indices=idx[0], grid=None, waste_type=waste_type)
        else:
            bins = None
    else:
        bins = None
        coord_size = 2 if method != 'triple' else 3
        depot = np.random.uniform(size=(dataset_size, coord_size))
        loc = np.random.uniform(size=(dataset_size, vrpp_size, coord_size))

    fill_values = []
    for _ in range(num_days):
        waste = generate_waste_prize(vrpp_size, distribution, (depot, loc), dataset_size, bins)
        fill_values.append(waste)
    
    fill_values = np.transpose(np.array(fill_values), (1, 0, 2))
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        fill_values.tolist() if fill_values.shape[0] > 1 else fill_values[0].tolist(),
        np.full(dataset_size, MAX_WASTE).tolist()  # Max waste before bin overflow, same for whole dataset
    ))


def generate_wcvrp_data(dataset_size, wcvrp_size, waste_type, 
                       distribution="gamma1", area="Rio Maior", focus_graph=None, focus_size=0, method=None, num_days=1):
    if focus_graph is not None:
        assert focus_size > 0
        depot, loc, mm_arr, idx = load_focus_coords(wcvrp_size, method, area, waste_type, focus_graph, focus_size)
        remaining_coords_size = dataset_size - focus_size
        random_coords = np.random.uniform(mm_arr[0], mm_arr[1], size=(remaining_coords_size, wcvrp_size+1, mm_arr.shape[-1]))
        depots, locs = process_coordinates(random_coords, method, col_names=None)
        depot = np.concatenate((depot, depots))
        loc = np.concatenate((loc, locs))
        assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]
        if distribution == 'emp':
            data_dir = get_path_until_string(focus_graph, 'wsr_simulator')
            bins = Bins(wcvrp_size, data_dir, sample_dist=distribution, area=area, indices=idx[0], grid=None, waste_type=waste_type)
        else:
            bins = None
    else:
        bins = None
        coord_size = 2 if method != 'triple' else 3
        depot = np.random.uniform(size=(dataset_size, coord_size))
        loc = np.random.uniform(size=(dataset_size, wcvrp_size, coord_size))

    # Bin waste fill at the start
    fill_values = []
    for _ in range(num_days):
        waste = generate_waste_prize(wcvrp_size, distribution, (depot, loc), dataset_size, bins)
        fill_values.append(waste)

    fill_values = np.transpose(np.array(fill_values), (1, 0, 2))
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        fill_values.tolist() if fill_values.shape[0] > 1 else fill_values[0].tolist(),
        np.full(dataset_size, MAX_WASTE).tolist()  # Max waste before bin overflow, same for whole dataset
    ))