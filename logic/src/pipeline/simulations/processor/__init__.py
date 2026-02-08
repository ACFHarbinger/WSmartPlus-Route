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
    """Sort dataframe.

    Args:
    df (Any): Description of df.
    metric_tosort (Any): Description of metric_tosort.
    ascending_order (Any): Description of ascending_order.

    Returns:
        Any: Description of return value.
    """
    return _mapper.sort_dataframe(df, metric_tosort, ascending_order)


def get_df_types(df, prec="32"):
    """Get df types.

    Args:
    df (Any): Description of df.
    prec (Any): Description of prec.

    Returns:
        Any: Description of return value.
    """
    return _mapper.get_df_types(df, prec)


def setup_df(depot, df, col_names, index_name="#bin"):
    """Setup df.

    Args:
    depot (Any): Description of depot.
    df (Any): Description of df.
    col_names (Any): Description of col_names.
    index_name (Any): Description of index_name.

    Returns:
        Any: Description of return value.
    """
    return _mapper.setup_df(depot, df, col_names, index_name)


def sample_df(df, n_elems, depot=None, output_path=None):
    """Sample df.

    Args:
    df (Any): Description of df.
    n_elems (Any): Description of n_elems.
    depot (Any): Description of depot.
    output_path (Any): Description of output_path.

    Returns:
        Any: Description of return value.
    """
    return _mapper.sample_df(df, n_elems, depot, output_path)


def process_indices(df, indices):
    """Process indices.

    Args:
    df (Any): Description of df.
    indices (Any): Description of indices.

    Returns:
        Any: Description of return value.
    """
    return _mapper.process_indices(df, indices)


def process_data(data, bins_coordinates, depot, indices=None):
    """Process data.

    Args:
    data (Any): Description of data.
    bins_coordinates (Any): Description of bins_coordinates.
    depot (Any): Description of depot.
    indices (Any): Description of indices.

    Returns:
        Any: Description of return value.
    """
    return _mapper.process_raw_data(data, bins_coordinates, depot, indices)


def process_coordinates(coords, method, col_names=["Lat", "Lng"]):
    """Process coordinates.

    Args:
    coords (Any): Description of coords.
    method (Any): Description of method.
    col_names (Any): Description of col_names.

    Returns:
        Any: Description of return value.
    """
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
    """Process model data.

    Args:
    coordinates (Any): Description of coordinates.
    dist_matrix (Any): Description of dist_matrix.
    device (Any): Description of device.
    method (Any): Description of method.
    configs (Any): Description of configs.
    edge_threshold (Any): Description of edge_threshold.
    edge_method (Any): Description of edge_method.
    area (Any): Description of area.
    waste_type (Any): Description of waste_type.
    adj_matrix (Any): Description of adj_matrix.

    Returns:
        Any: Description of return value.
    """
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
    """Create dataframe from matrix.

    Args:
    matrix (Any): Description of matrix.

    Returns:
        Any: Description of return value.
    """
    enchimentos = [row[-1] for row in matrix]
    ids_rota = np.arange(len(matrix))
    return pd.DataFrame({"#bin": ids_rota, "Stock": enchimentos, "Accum_Rate": np.zeros(len(ids_rota))})


def convert_to_dict(bins_coordinates):
    """Convert to dict.

    Args:
    bins_coordinates (Any): Description of bins_coordinates.

    Returns:
        Any: Description of return value.
    """
    coordinates_dict = {}
    for _, row in bins_coordinates.iterrows():
        coordinates_dict[row["ID"]] = {"lat": np.float64(row["Lat"]), "lng": np.float64(row["Lng"])}
    return coordinates_dict


def save_matrix_to_excel(matrix, results_dir, seed, data_dist, policy, sample_id):
    """Save matrix to excel.

    Args:
    matrix (Any): Description of matrix.
    results_dir (Any): Description of results_dir.
    seed (Any): Description of seed.
    data_dist (Any): Description of data_dist.
    policy (Any): Description of policy.
    sample_id (Any): Description of sample_id.

    Returns:
        Any: Description of return value.
    """
    return _mapper.save_results(matrix, results_dir, seed, data_dist, policy, sample_id)


def setup_basedata(n_bins, data_dir, area, waste_type):
    """Setup basedata.

    Args:
    n_bins (Any): Description of n_bins.
    data_dir (Any): Description of data_dir.
    area (Any): Description of area.
    waste_type (Any): Description of waste_type.

    Returns:
        Any: Description of return value.
    """
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
    """Setup dist path tup.

    Args:
    bins_coordinates (Any): Description of bins_coordinates.
    size (Any): Description of size.
    dist_method (Any): Description of dist_method.
    dm_filepath (Any): Description of dm_filepath.
    env_filename (Any): Description of env_filename.
    gapik_file (Any): Description of gapik_file.
    symkey_name (Any): Description of symkey_name.
    device (Any): Description of device.
    edge_thresh (Any): Description of edge_thresh.
    edge_method (Any): Description of edge_method.
    focus_idx (Any): Description of focus_idx.

    Returns:
        Any: Description of return value.
    """
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
