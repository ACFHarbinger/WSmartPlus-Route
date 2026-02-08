"""loader.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import loader
    """
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from tensordict import TensorDict

from logic.src.utils.functions import get_path_until_string


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    Filters out None values from samples.

    Args:
        batch: List of samples.

    Returns:
        Collated batch.
    """
    batch = [{key: val for key, val in sample.items() if val is not None} for sample in batch if sample is not None]

    # Empty lists can break collate
    if len(batch) == 0:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)


def load_focus_coords(
    graph_size: int,
    method: Optional[str],
    area: str,
    waste_type: str,
    focus_graph: str,
    focus_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Loads coordinates and depot information from simulator data.

    Args:
        graph_size: Size of the graph.
        method: Method for coordinate processing (e.g., 'naive', 'k-means').
        area: Geographic area name.
        waste_type: Type of waste.
        focus_graph: Path to focus graph file.
        focus_size: Replication factor for focus graph. Defaults to 1.

    Returns:
        Tuple of (depot, locations, minmax array, index).
    """
    # Lazy imports to avoid circular dependency
    from logic.src.pipeline.simulations.processor import process_coordinates, process_data
    from logic.src.pipeline.simulations.repository import load_depot, load_simulator_data

    focus_graph_dir = get_path_until_string(focus_graph, "wsr_simulator")
    if focus_graph_dir is None:
        raise ValueError(f"Could not find 'wsr_simulator' in path {focus_graph}")

    depot = load_depot(focus_graph_dir, area)
    data, coords = load_simulator_data(focus_graph_dir, graph_size, area, waste_type)
    with open(focus_graph) as js:
        idx = json.load(js)

    _, coords = process_data(data, coords, depot, idx[0])
    if method is None:
        return coords, idx, None, None  # type: ignore

    depot, loc = process_coordinates(coords, method)
    if focus_size > 0:
        lat_minmax = (coords["Lat"].min(), coords["Lat"].max())
        lng_minmax = (coords["Lng"].min(), coords["Lng"].max())
        mm_arr = np.array([lng_minmax, lat_minmax])
        ret_val = (
            np.tile(depot, (focus_size, 1)),
            np.tile(loc, (focus_size, 1, 1)),
            mm_arr.T,
            idx,
        )
    else:
        ret_val = (depot, loc, None, idx)  # type: ignore
    return ret_val  # type: ignore


def check_extension(filename: str, extension: str = ".pkl") -> str:
    """
    Ensures filename has the specified extension.

    Args:
        filename: Input filename.
        extension: Desired extension (e.g., '.pkl', '.td', '.pt').

    Returns:
        Filename with the specified extension.
    """
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def save_dataset(dataset: Any, filename: str) -> None:
    """
    Saves a dataset using pickle.

    Args:
        dataset: The data to save.
        filename: Target filename.

    Raises:
        Exception: If directory creation fails.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        try:
            os.makedirs(filedir, exist_ok=True)
        except Exception:
            raise Exception("directories to save datasets do not exist and could not be created")

    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


def save_td_dataset(td: TensorDict, filename: str) -> None:
    """
    Saves a TensorDict dataset.

    Args:
        td: The TensorDict to save.
        filename: Target filename.
    """
    filedir = os.path.split(filename)[0]
    if filedir and not os.path.isdir(filedir):
        os.makedirs(filedir, exist_ok=True)

    torch.save(td, check_extension(filename, ".td"))


def load_td_dataset(filename: str, device: str = "cpu") -> TensorDict:
    """
    Loads a TensorDict dataset.

    Args:
        filename: The filename.
        device: Device to load onto.

    Returns:
        The loaded TensorDict.
    """
    return torch.load(check_extension(filename, ".td"), map_location=device)


def load_dataset(filename: str) -> Any:
    """
    Loads a dataset from a pickle file.

    Args:
        filename: The filename.

    Returns:
        The loaded dataset.
    """
    with open(check_extension(filename, ".pkl"), "rb") as f:
        return pickle.load(f)
