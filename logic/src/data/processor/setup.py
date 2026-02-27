"""Simulation setup functions for base data and distance computation."""

import contextlib

import numpy as np
import torch

from logic.src.data.network import compute_distance_matrix
from logic.src.pipeline.simulations.repository import load_depot, load_simulator_data
from logic.src.utils.graph.network_utils import apply_edges, get_paths_between_states

from ._logging import _log_processor_event


def setup_basedata(n_bins, data_dir, area, waste_type):
    """Load depot, bin statistics and coordinates for a given area.

    Args:
        n_bins: Number of bins.
        data_dir: Base data directory.
        area: Geographic area name.
        waste_type: Waste type identifier.

    Returns:
        Tuple of (data, bins_coordinates, depot) DataFrames.
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
    """Compute distance matrix, edge structure and shortest paths.

    Args:
        bins_coordinates: Coordinates DataFrame.
        size: Number of customer nodes (excludes depot).
        dist_method: Distance computation method.
        dm_filepath: Optional pre-computed distance matrix file.
        env_filename: Environment config file.
        gapik_file: Google API key file.
        symkey_name: Symmetric key name.
        device: Torch device.
        edge_thresh: Edge filtering threshold.
        edge_method: Edge generation method.
        focus_idx: Optional focus indices.

    Returns:
        Tuple of ((dist_matrix_edges, paths, dm_tensor, distC), adj_matrix).
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
