import os
import json
import math
import torch
import pickle
import numpy as np

from logic.src.utils.functions import get_path_until_string
from logic.src.pipeline.simulator.loader import load_depot, load_simulator_data
from logic.src.pipeline.simulator.processor import process_data, process_coordinates


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        try:
            os.makedirs(filedir, exist_ok=True)
        except Exception:
            raise Exception("directories to save datasets do not exist and could not be created")

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def collate_fn(batch):
    #batch = [{key: val if val is not None else torch.empty(1) for key, val in sample.items()} for sample in batch]
    batch = [{key: val for key, val in sample.items() if val is not None} for sample in batch]

    # Empty lists can break collate
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)


def load_focus_coords(graph_size, method, area, waste_type, focus_graph, focus_size=1):
    focus_graph_dir = get_path_until_string(focus_graph, 'wsr_simulator')
    depot = load_depot(focus_graph_dir, area)
    data, coords = load_simulator_data(focus_graph_dir, graph_size, area, waste_type)
    with open(os.path.join(focus_graph)) as js:
        idx = json.load(js)

    _, coords = process_data(data, coords, depot, idx[0])
    if method is None:
        return coords, idx

    depot, loc = process_coordinates(coords, method)
    if focus_size > 0:
        lat_minmax = (coords['Lat'].min(), coords['Lat'].max())
        lng_minmax = (coords['Lng'].min(), coords['Lng'].max())
        mm_arr = np.array([lng_minmax, lat_minmax])
        ret_val = (np.tile(depot, (focus_size, 1)), np.tile(loc, (focus_size, 1, 1)), mm_arr.T, idx)
    else:
        ret_val = (depot, loc, None, idx)
    #depot = np.array([coords['Lng'].iloc[0], coords['Lat'].iloc[0]])
    #loc = np.array([[x, y] for x, y in zip(coords['Lng'].iloc[1:], coords['Lat'].iloc[1:])])
    return ret_val


def _get_fill_gamma(dataset_size, problem_size, gamma_option):
    def __set_distribution_param(size, param):
        param_len = len(param)
        if size == param_len:
            return param
        
        param = param * math.ceil(size / param_len)
        if size % param_len != 0:
            param = param[:param_len-size % param_len]
        return param

    if gamma_option == 0:
        alpha = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10]
        theta = [5, 2]
    elif gamma_option == 1:
        alpha = [2, 2, 2, 2, 2, 6, 6, 6, 6, 6]
        theta = [6, 4]
    elif gamma_option == 2:
        alpha = [1, 1, 1, 1, 1, 3, 3, 3, 3, 3]
        theta = [8, 6]
    else:
        assert gamma_option == 3
        alpha = [5, 2]
        theta = [10]

    k = __set_distribution_param(problem_size, alpha)
    th = __set_distribution_param(problem_size, theta)
    return np.random.gamma(k, th, size=(dataset_size, problem_size)) / 100.


def generate_waste_prize(problem_size, distribution, graph, dataset_size=1, bins=None):
    if distribution == 'empty':
        wp = np.zeros(shape=(dataset_size, problem_size))
    elif distribution == 'const':
        wp = np.ones(shape=(dataset_size, problem_size))
    elif distribution == 'unif':
        wp = (1 + np.random.randint(0, 100, size=(dataset_size, problem_size))) / 100.
    elif 'gamma' in distribution:
        gamma_option = int(distribution[-1]) - 1
        wp = _get_fill_gamma(dataset_size, problem_size, gamma_option)
    elif 'emp' in distribution:
        wp = bins.stochasticFilling(n_samples=dataset_size, only_fill=True) / 100.
    else:
        assert distribution == 'dist'
        depot, loc = graph
        if dataset_size > 1:
            wp = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
            return (1 + (wp_ / wp_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.
        else:
            wp_ = (depot[None, :] - loc).norm(p=2, dim=-1)
            return (1 + (wp_ / wp_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    if dataset_size == 1:
        return wp[0]
    return wp
