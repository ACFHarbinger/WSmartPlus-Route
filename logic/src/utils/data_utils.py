from __future__ import annotations

import json
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from logic.src.pipeline.simulations.loader import load_depot, load_simulator_data
from logic.src.pipeline.simulations.processor import process_coordinates, process_data
from logic.src.utils.functions.function import get_path_until_string


def check_extension(filename: str) -> str:
    """
    Ensures filename has .pkl extension.

    Args:
        filename: Input filename.

    Returns:
        Filename with .pkl extension.
    """
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
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

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename: str) -> Any:
    """
    Loads a dataset from a pickle file.

    Args:
        filename: The filename.

    Returns:
        The loaded dataset.
    """
    with open(check_extension(filename), "rb") as f:
        return pickle.load(f)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    Filters out None values from samples.

    Args:
        batch: List of samples.

    Returns:
        Collated batch.
    """
    batch = [{key: val for key, val in sample.items() if val is not None} for sample in batch]

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


def _get_fill_gamma(dataset_size: int, problem_size: int, gamma_option: int) -> np.ndarray:
    """
    Generates waste levels using a Gamma distribution.

    Args:
        dataset_size: Number of instances.
        problem_size: Number of bins per instance.
        gamma_option: Index to select gamma parameters (alpha, theta).

    Returns:
        Generated waste levels [dataset_size, problem_size].
    """

    def __set_distribution_param(size: int, param: List[int]) -> List[int]:
        """
        Sets a distribution parameter if not already present.

        Args:
            size: Size to match.
            param: Parameter list.

        Returns:
            Adjusted parameter list.
        """
        param_len = len(param)
        if size == param_len:
            return param

        param = param * math.ceil(size / param_len)
        if size % param_len != 0:
            param = param[: param_len - size % param_len]
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
    return np.random.gamma(k, th, size=(dataset_size, problem_size)) / 100.0


def generate_waste_prize(
    problem_size: int,
    distribution: str,
    graph: Tuple[Any, Any],
    dataset_size: int = 1,
    bins: Optional[Any] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generates waste or prize values based on a distribution or empirical data.

    Args:
        problem_size: Number of nodes/bins.
        distribution: Distribution type ('empty', 'const', 'unif', 'gammaX', 'emp', 'dist').
        graph: (depot, loc) coordinates.
        dataset_size: Number of datasets to generate. Defaults to 1.
        bins: Bins object for empirical sampling.

    Returns:
        Generated waste/prizes.
    """
    if distribution == "empty":
        wp: Any = np.zeros(shape=(dataset_size, problem_size))
    elif distribution == "const":
        wp = np.ones(shape=(dataset_size, problem_size))
    elif distribution == "unif":
        wp = (1 + np.random.randint(0, 100, size=(dataset_size, problem_size))) / 100.0
    elif "gamma" in distribution:
        gamma_option = int(distribution[-1]) - 1
        wp = _get_fill_gamma(dataset_size, problem_size, gamma_option)
    elif "emp" in distribution:
        if bins is None:
            raise ValueError("bins must be provided for empirical distribution")
        wp = bins.stochasticFilling(n_samples=dataset_size, only_fill=True) / 100.0
    else:
        assert distribution == "dist"
        depot, loc = graph
        if dataset_size > 1:
            wp = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
            return (1 + (wp / wp.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.0
        else:
            wp_float = (depot[None, :] - loc).norm(p=2, dim=-1)
            return (1 + (wp_float / wp_float.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.0

    if dataset_size == 1:
        return wp[0]
    return wp
