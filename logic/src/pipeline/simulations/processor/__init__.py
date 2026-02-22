"""
Data Processing and Transformation Pipeline for Simulation Setup.
"""

import contextlib

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


def _log_processor_event(event_name, variable_name="data", event_type="mutate", shape=None, **kwargs):
    import sys

    try:
        source_line = sys._getframe(1).f_lineno
    except Exception:
        source_line = 0

    with contextlib.suppress(Exception):
        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is not None:
            metadata = {
                "event": event_name,
                "variable_name": variable_name,
                "source_file": "processor/__init__.py",
                "source_line": source_line,
            }
            metadata.update(kwargs)
            safe_meta = {}
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            run.log_dataset_event(event_type, shape=shape, metadata=safe_meta)


def sort_dataframe(df, metric_tosort, ascending_order=True):
    """Sort dataframe.

    Args:
    df (Any): Description of df.
    metric_tosort (Any): Description of metric_tosort.
    ascending_order (Any): Description of ascending_order.

    Returns:
        Any: Description of return value.
    """
    result = _mapper.sort_dataframe(df, metric_tosort, ascending_order)
    _log_processor_event("sort_dataframe", variable_name="dataframe", metric=metric_tosort, ascending=ascending_order)
    return result


def get_df_types(df, prec="32"):
    """Get df types.

    Args:
    df (Any): Description of df.
    prec (Any): Description of prec.

    Returns:
        Any: Description of return value.
    """
    result = _mapper.get_df_types(df, prec)
    _log_processor_event("get_df_types", variable_name="df_types", prec=prec)
    return result


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
    result = _mapper.setup_df(depot, df, col_names, index_name)
    _log_processor_event("setup_df", variable_name="dataframe", col_names=col_names, index_name=index_name)
    return result


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
    result = _mapper.sample_df(df, n_elems, depot, output_path)
    _log_processor_event("sample_df", variable_name="dataframe", n_elems=n_elems, output_path=output_path)
    return result


def process_indices(df, indices):
    """Process indices.

    Args:
    df (Any): Description of df.
    indices (Any): Description of indices.

    Returns:
        Any: Description of return value.
    """
    result = _mapper.process_indices(df, indices)
    _log_processor_event(
        "process_indices", variable_name="indices", n_indices=len(indices) if indices is not None else 0
    )
    return result


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
    result = _mapper.process_raw_data(data, bins_coordinates, depot, indices)
    with contextlib.suppress(Exception):
        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is not None:
            run.log_params({"data.n_bins_after_index_filter": len(result[0])})

    _log_processor_event("process_data", variable_name="data", n_bins=len(result[0]))
    return result


def process_coordinates(coords, method, col_names=None):
    """Process coordinates.

    Args:
    coords (Any): Description of coords.
    method (Any): Description of method.
    col_names (Any): Description of col_names.

    Returns:
        Any: Description of return value.
    """
    if col_names is None:
        col_names = ["Lat", "Lng"]
    result = format_coordinates(coords, method, col_names)
    _log_processor_event("process_coordinates", variable_name="coordinates", method=method, col_names=col_names)
    return result


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
    result = _mapper.process_model_input(
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
    with contextlib.suppress(Exception):
        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is not None:
            run.log_params(
                {
                    "data.problem_size": int(len(dist_matrix) - 1),
                    "data.coord_method": str(method),
                    "data.has_edge_filter": bool(0 < edge_threshold < 1),
                    "data.edge_threshold": float(edge_threshold),
                    "data.edge_method": str(edge_method),
                }
            )

    _log_processor_event("process_model_data", variable_name="model_input", problem_size=int(len(dist_matrix) - 1))
    return result


def create_dataframe_from_matrix(matrix):
    """Create dataframe from matrix.

    Args:
    matrix (Any): Description of matrix.

    Returns:
        Any: Description of return value.
    """
    enchimentos = [row[-1] for row in matrix]
    ids_rota = np.arange(len(matrix))
    result = pd.DataFrame({"#bin": ids_rota, "Stock": enchimentos, "Accum_Rate": np.zeros(len(ids_rota))})
    _log_processor_event("create_dataframe_from_matrix", variable_name="dataframe", num_rows=len(matrix))
    return result


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
    _log_processor_event("convert_to_dict", variable_name="coordinates_dict", num_bins=len(bins_coordinates))
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
    result = _mapper.save_results(matrix, results_dir, seed, data_dist, policy, sample_id)
    _log_processor_event(
        "save_matrix_to_excel",
        variable_name="excel_file",
        event_type="save",
        seed=seed,
        data_dist=data_dist,
        policy=policy,
        sample_id=sample_id,
    )
    return result


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
    _log_processor_event("setup_basedata", variable_name="base_data", n_bins=n_bins, area=area, waste_type=waste_type)
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
    print(dist_matrix.shape)
    dist_matrix_edges, shortest_paths, adj_matrix = apply_edges(dist_matrix, edge_thresh, edge_method)
    paths = get_paths_between_states(size + 1, shortest_paths)
    dm_tensor = torch.from_numpy(dist_matrix_edges / 100.0).to(device)
    distC = np.round(dist_matrix_edges * 10).astype("int32")

    with contextlib.suppress(Exception):
        from logic.src.tracking.core.run import get_active_run

        run = get_active_run()
        if run is not None:
            n_nonzero = int(np.count_nonzero(dist_matrix_edges))
            n_total = int(dist_matrix_edges.size)
            run.log_params(
                {
                    "data.edge_threshold": float(edge_thresh),
                    "data.edge_method": str(edge_method),
                    "data.n_nodes": int(size + 1),
                }
            )
            run.log_metric("data/n_edges", float(n_nonzero))
            run.log_metric("data/edge_density", float(n_nonzero) / float(n_total) if n_total > 0 else 0.0)

    _log_processor_event(
        "setup_dist_path_tup",
        variable_name="distance_matrix",
        size=size,
        dist_method=dist_method,
        edge_thresh=edge_thresh,
    )
    return (dist_matrix_edges, paths, dm_tensor, distC), adj_matrix
