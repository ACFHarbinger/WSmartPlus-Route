import torch
import numpy as np
import scipy.stats as stats

from logic.src.utils.definitions import MAX_WASTE, MAX_LENGTHS, CAPACITIES
from logic.src.utils.functions import get_path_until_string
from logic.src.utils.data_utils import generate_waste_prize, load_focus_coords
from logic.src.pipeline.simulator.bins import Bins
from logic.src.pipeline.simulator.processor import process_coordinates


__all__ = [
    'generate_pdp_data',
    'generate_tsp_data',
    'generate_vrp_data',
    'generate_op_data',
    'generate_vrpp_data',
    'generate_wcrp_data',
    'generate_pctsp_data'
]


def generate_pdp_data(dataset_size, pdp_size, is_gaussian, sigma, area="Rio Maior", focus_graph=None, focus_size=0, method=None):
    if is_gaussian:
        def truncated_normal(graph_size, sigma):
            mu = 0.5
            lower, upper = 0, 1
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            return torch.stack([torch.from_numpy(X.rvs(graph_size)), torch.from_numpy(X.rvs(graph_size))], 1).tolist()

        def generate(dataset_size, graph_size):
            data = []
            for i in range(dataset_size):
                data.append(truncated_normal(graph_size, sigma))

            return data
        return list(zip(truncated_normal(dataset_size, sigma),  # Depot location
                        generate(dataset_size, pdp_size)
                ))
    else:
        return list(zip(np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
                        np.random.uniform(size=(dataset_size, pdp_size, 2)).tolist()
                    ))
    

def generate_tsp_data(dataset_size, tsp_size, area="Rio Maior", focus_graph=None, focus_size=0, method=None):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size, area="Rio Maior", focus_graph=None, focus_size=0, method=None):
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_op_data(dataset_size, op_size, prize_type='const', area="Rio Maior", focus_graph=None, focus_size=0, method=None):
    if focus_graph is not None:
        assert focus_size > 0
        depot, loc, mm_arr, _ = load_focus_coords(op_size, method, area, prize_type, focus_graph, focus_size)
        remaining_coords_size = dataset_size - focus_size
        random_coords = np.random.uniform(mm_arr[0], mm_arr[1], size=(remaining_coords_size, op_size+1, mm_arr.shape[-1]))
        depots, locs = process_coordinates(random_coords, method, col_names=None)
        depot = np.concatenate((depot, depots))
        loc = np.concatenate((loc, locs))
        assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]
    else:
        coord_size = 2 if method != 'triple' else 3
        depot = np.random.uniform(size=(dataset_size, coord_size))
        loc = np.random.uniform(size=(dataset_size, op_size, coord_size))

    prize = generate_waste_prize(op_size, prize_type, (depot, loc), dataset_size)

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


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


def generate_wcrp_data(dataset_size, wcrp_size, waste_type, 
                       distribution="gamma1", area="Rio Maior", focus_graph=None, focus_size=0, method=None, num_days=1):
    if focus_graph is not None:
        assert focus_size > 0
        depot, loc, mm_arr, idx = load_focus_coords(wcrp_size, method, area, waste_type, focus_graph, focus_size)
        remaining_coords_size = dataset_size - focus_size
        random_coords = np.random.uniform(mm_arr[0], mm_arr[1], size=(remaining_coords_size, wcrp_size+1, mm_arr.shape[-1]))
        depots, locs = process_coordinates(random_coords, method, col_names=None)
        depot = np.concatenate((depot, depots))
        loc = np.concatenate((loc, locs))
        assert depot.shape[-1] == loc.shape[-1] and depot.shape[0] == loc.shape[0]
        if distribution == 'emp':
            data_dir = get_path_until_string(focus_graph, 'wsr_simulator')
            bins = Bins(wcrp_size, data_dir, sample_dist=distribution, area=area, indices=idx[0], grid=None, waste_type=waste_type)
        else:
            bins = None
    else:
        bins = None
        coord_size = 2 if method != 'triple' else 3
        depot = np.random.uniform(size=(dataset_size, coord_size))
        loc = np.random.uniform(size=(dataset_size, wcrp_size, coord_size))

    # Bin waste fill at the start
    fill_values = []
    for _ in range(num_days):
        waste = generate_waste_prize(wcrp_size, distribution, (depot, loc), dataset_size, bins)
        fill_values.append(waste)

    fill_values = np.transpose(np.array(fill_values), (1, 0, 2))
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        fill_values.tolist() if fill_values.shape[0] > 1 else fill_values[0].tolist(),
        np.full(dataset_size, MAX_WASTE).tolist()  # Max waste before bin overflow, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3, area="Rio Maior", focus_graph=None, focus_size=0, method=None):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2
    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))