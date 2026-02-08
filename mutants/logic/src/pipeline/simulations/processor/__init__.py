"""
Data Processing and Transformation Pipeline for Simulation Setup.
"""

import numpy as np
import pandas as pd
import torch

from ....utils.graph.network_utils import apply_edges, get_paths_between_states
from ..network import compute_distance_matrix
from ..repository import load_depot, load_simulator_data
from .formatting import format_coordinates
from .mapper import SimulationDataMapper

__all__ = [
    "SimulationDataMapper",
    "format_coordinates",
    "sort_dataframe",
    "get_df_types",
    "setup_df",
    "sample_df",
    "process_indices",
    "process_data",
    "process_coordinates",
    "process_model_data",
    "create_dataframe_from_matrix",
    "convert_to_dict",
    "save_matrix_to_excel",
    "setup_basedata",
    "setup_dist_path_tup",
]

_mapper = SimulationDataMapper()


def sort_dataframe(df, metric_tosort, ascending_order=True):
    return _mapper.sort_dataframe(df, metric_tosort, ascending_order)


def get_df_types(df, prec="32"):
    return _mapper.get_df_types(df, prec)


def setup_df(depot, df, col_names, index_name="#bin"):
    return _mapper.setup_df(depot, df, col_names, index_name)


def sample_df(df, n_elems, depot=None, output_path=None):
    return _mapper.sample_df(df, n_elems, depot, output_path)


def process_indices(df, indices):
    return _mapper.process_indices(df, indices)


def process_data(data, bins_coordinates, depot, indices=None):
    return _mapper.process_raw_data(data, bins_coordinates, depot, indices)


def process_coordinates(coords, method, col_names=["Lat", "Lng"]):
    return format_coordinates(coords, method, col_names)


def process_model_data(
    coordinates,
    dist_matrix,
    device,
    method,
    configs,
    edge_threshold,
    edge_method,
    area,
    waste_type,
    adj_matrix=None,
):
    return _mapper.process_model_input(
        coordinates,
        dist_matrix,
        device,
        method,
        configs,
        edge_threshold,
        edge_method,
        area,
        waste_type,
        adj_matrix,
    )


def create_dataframe_from_matrix(matrix):
    enchimentos = [row[-1] for row in matrix]
    ids_rota = np.arange(len(matrix))
    return pd.DataFrame({"#bin": ids_rota, "Stock": enchimentos, "Accum_Rate": np.zeros(len(ids_rota))})


def convert_to_dict(bins_coordinates):
    coordinates_dict = {}
    for _, row in bins_coordinates.iterrows():
        coordinates_dict[row["ID"]] = {"lat": np.float64(row["Lat"]), "lng": np.float64(row["Lng"])}
    return coordinates_dict


def save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id):
    return _mapper.save_results(matrix, results_dir, seed, data_dist, policy, sample_id)


def setup_basedata(n_bins, data_dir, area, waste_type):
    depot = load_depot(data_dir, area)
    data, bins_coordinates = load_simulator_data(data_dir, n_bins, area, waste_type)
    return data, bins_coordinates, depot


def setup_dist_path_tup(
    bins_coordinates,
    size,
    dist_method,
    dm_filepath,
    env_filename,
    gapik_file,
    symkey_name,
    device,
    edge_thresh,
    edge_method,
    focus_idx=None,
):
    dist_matrix = compute_distance_matrix(
        bins_coordinates,
        dist_method,
        dm_filepath=dm_filepath,
        env_filename=env_filename,
        gapik_file=gapik_file,
        symkey_name=symkey_name,
        focus_idx=focus_idx,
    )
    dist_matrix_edges, shortest_paths, adj_matrix = apply_edges(dist_matrix, edge_thresh, edge_method)
    paths = get_paths_between_states(size + 1, shortest_paths)
    dm_tensor = torch.from_numpy(dist_matrix_edges / 100.0).to(device)
    distC = np.round(dist_matrix_edges * 10).astype("int32")
    return (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix
