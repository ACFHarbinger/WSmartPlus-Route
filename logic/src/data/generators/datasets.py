"""datasets.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import datasets
"""

import os
import random
from typing import Any, Optional

import numpy as np
import torch

from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.data import DataConfig
from logic.src.constants import ROOT_DIR
from logic.src.data.generators.builders import VRPInstanceBuilder
from logic.src.pipeline.simulations.repository import FileSystemRepository, set_repository
from logic.src.utils.data.data_utils import (
    check_extension,
    save_simulation_dataset,
    save_td_dataset,
)


def generate_datasets(cfg: Config) -> None:
    """
    Generates VRP datasets based on the typed Hydra configuration.

    Args:
        cfg: Root Hydra configuration with ``cfg.data`` containing data
            generation parameters.
    """
    from logic.src.data.generators.validators import validate_data_config

    # Validate and sanitize config values
    validate_data_config(cfg)

    # Initialize the filesystem repository for coordinate/depot loading
    set_repository(FileSystemRepository(ROOT_DIR))

    data = cfg.data

    # Set the random seed and execute the program
    random.seed(data.seed)
    np.random.seed(data.seed)
    torch.manual_seed(data.seed)

    gamma_dists = ["gamma1", "gamma2", "gamma3", "gamma4"]
    distributions_per_problem = {
        "vrpp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "wcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "swcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
    }

    # Define the problem distribution(s)
    if data.problem == "all":
        problems = distributions_per_problem
    else:
        problems = {
            data.problem: (
                distributions_per_problem[data.problem]
                if len(data.data_distributions) == 1 and data.data_distributions[0] == "all"
                else list(data.data_distributions)
            )
        }

    for problem, distributions in problems.items():
        _generate_problem_data(problem, distributions, data)


def _generate_problem_data(problem: str, distributions: Any, data: DataConfig) -> None:
    """Helper to generate data for a specific problem and its distributions."""
    datadir = (
        os.path.join(ROOT_DIR, "data", data.data_dir, problem)
        if data.dataset_type in ["train_time", "train"]
        else os.path.join(ROOT_DIR, "data", "wsr_simulator", data.data_dir)
    )
    try:
        os.makedirs(datadir, exist_ok=True)
    except Exception as e:
        raise Exception("directories to save generated data files do not exist and could not be created") from e

    graphs = list(data.graphs) if data.graphs else []

    try:
        for dist in distributions or [None]:
            if graphs:
                for graph_cfg in graphs:
                    _process_instance_generation(problem, dist, datadir, data, graph_cfg=graph_cfg)
            else:
                print("[WARNING] No graphs provided for instance generation. Skipping.")
    except Exception as e:
        has_dists = len(distributions) >= 1 and distributions[0] is not None
        raise Exception(
            "failed to generate data for problem {}{}  due to {}".format(
                problem, f" {distributions}" if has_dists else "", repr(e)
            )
        ) from e


def _process_instance_generation(
    problem: str,
    dist: Any,
    datadir: str,
    data: DataConfig,
    graph_cfg: Optional[GraphConfig] = None,
) -> None:
    """Configure builder and save datasets for a specific configuration."""
    if graph_cfg is not None:
        size = graph_cfg.num_loc
        graph = graph_cfg.focus_graph
        area = graph_cfg.area
        waste_type = graph_cfg.waste_type
        vertex_method = graph_cfg.vertex_method
        focus_size = graph_cfg.focus_size if graph_cfg.focus_size is not None else 31
        n_days = graph_cfg.n_days if graph_cfg.n_days is not None else max(data.n_epochs - data.epoch_start, 1)
        n_samples = graph_cfg.n_samples if graph_cfg.n_samples is not None else data.dataset_size
    else:
        size = 50
        graph = None
        area = "riomaior"
        waste_type = "plastic"
        vertex_method = "mmn"
        focus_size = 31
        n_days = max(data.n_epochs - data.epoch_start, 1)
        n_samples = data.dataset_size

    # dataset_type == "train" implies 0 days in legacy logic
    if data.dataset_type == "train":
        n_days = 0

    if graph is not None and not os.path.isfile(graph):
        sim_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        if os.path.isfile(os.path.join(sim_dir, graph)):
            graph = os.path.join(sim_dir, graph)
        elif os.path.isfile(os.path.join(sim_dir, "bins_selection", graph)):
            graph = os.path.join(sim_dir, "bins_selection", graph)

    assert size is not None, "size must be provided"
    assert waste_type is not None, "waste_type must be provided"
    assert area is not None, "area must be provided"

    name = data.name or "dataset"
    print(
        "Generating '{}{}' ({}) dataset for the {} with {} locations{}{}".format(
            name,
            n_days if n_days > 0 else "",
            data.dataset_type or "train",
            problem.upper(),
            size,
            " and using '{}' as the instance distribution".format(dist),
            ":" if n_days == 0 else "...",
        )
    )

    builder = VRPInstanceBuilder()
    effective_focus_size = n_samples if data.dataset_type == "test_simulator" else focus_size
    builder.set_dataset_size(n_samples).set_problem_size(size).set_waste_type(waste_type).set_distribution(
        dist or "empty"
    ).set_area(area).set_focus_graph(graph, effective_focus_size).set_method(vertex_method).set_problem_name(problem)

    _apply_noise_config(builder, problem, data)

    if data.dataset_type == "test_simulator":
        _generate_test_simulator_data(builder, n_days, datadir, dist, size, data, area, waste_type, n_samples)
    elif data.dataset_type == "train_time":
        _generate_train_time_data(builder, problem, n_days, datadir, dist, size, data)
    else:
        _generate_train_data(builder, problem, datadir, dist, size, data)


def _apply_noise_config(builder: VRPInstanceBuilder, problem: str, data: DataConfig) -> None:
    """Apply noise parameters to the builder."""
    if data.mu is not None:
        sigma = data.sigma
        if isinstance(sigma, list):
            sigma = sigma[0]
        builder.set_noise(data.mu[0] if isinstance(data.mu, list) else data.mu, (sigma**2 if sigma > 0 else 0.0))
    elif problem == "swcvrp":
        sigma = data.sigma
        if isinstance(sigma, list):
            sigma = sigma[0]
        mu = data.mu[0] if isinstance(data.mu, list) and data.mu else 0.0
        builder.set_noise(
            mu if mu is not None else 0.0,
            (sigma**2 if sigma > 0 else 1.0),
        )


def _generate_test_simulator_data(
    builder: VRPInstanceBuilder,
    n_days: int,
    datadir: str,
    dist: Any,
    size: int,
    data: DataConfig,
    area: str,
    waste_type: str,
    n_samples: int,
) -> None:
    """Generate and save test simulator data."""
    name = data.name or "dataset"
    builder.set_num_days(n_days)
    if data.filename is None:
        filename = os.path.join(
            datadir,
            "{}{}{}_{}{}_N{}_seed{}.npz".format(
                area,
                size,
                "_{}".format(dist) if dist is not None else "",
                name,
                n_days if n_days > 0 else "",
                n_samples,
                data.seed,
            ),
        )
    else:
        filename = check_extension(data.filename, ".npz")

    _verify_and_save(builder, filename, data, is_td=False)


def _generate_train_time_data(
    builder: VRPInstanceBuilder, problem: str, n_days: int, datadir: str, dist: Any, size: int, data: DataConfig
) -> None:
    """Generate and save train time data."""
    name = data.name or "dataset"
    builder.set_num_days(data.n_epochs)
    if data.filename is None:
        ext = ".td"
        if data.mu is not None:
            mu_val = data.mu[0] if isinstance(data.mu, list) else data.mu
            sigma_val = data.sigma[0] if isinstance(data.sigma, list) else data.sigma
            filename = os.path.join(
                datadir,
                "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                    problem,
                    size,
                    "_{}".format(dist) if dist is not None else "",
                    name,
                    n_days if n_days > 0 else "",
                    data.seed,
                    f"gaussian{mu_val}",
                    f"_{sigma_val}",
                    ext,
                ),
            )
        else:
            filename = os.path.join(
                datadir,
                "{}{}{}_{}{}_seed{}{}".format(
                    problem,
                    size,
                    "_{}".format(dist) if dist is not None else "",
                    name,
                    n_days if n_days > 0 else "",
                    data.seed,
                    ext,
                ),
            )
    else:
        filename = check_extension(data.filename, ".td")

    _verify_and_save(builder, filename, data, is_td=True)


def _generate_train_data(
    builder: VRPInstanceBuilder, problem: str, datadir: str, dist: Any, size: int, data: DataConfig
) -> None:
    """Generate and save standard training data."""
    name = data.name or "dataset"
    builder.set_num_days(1)
    for epoch in range(data.epoch_start, data.n_epochs):
        print("- Generating epoch {} data".format(epoch))
        if data.filename is None:
            ext = ".td"
            if data.mu is not None:
                mu_val = data.mu[0] if isinstance(data.mu, list) else data.mu
                sigma_val = data.sigma[0] if isinstance(data.sigma, list) else data.sigma
                filename = os.path.join(
                    datadir,
                    "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                        problem,
                        size,
                        ("_{}".format(dist) if dist is not None else ""),
                        name,
                        epoch if data.n_epochs > 1 else "",
                        data.seed,
                        f"gaussian{mu_val}",
                        f"_{sigma_val}",
                        ext,
                    ),
                )
            else:
                filename = os.path.join(
                    datadir,
                    "{}{}{}_{}{}_seed{}{}".format(
                        problem,
                        size,
                        ("_{}".format(dist) if dist is not None else ""),
                        name,
                        epoch if data.n_epochs > 1 else "",
                        data.seed,
                        ext,
                    ),
                )
        else:
            filename = check_extension(data.filename, ".td")

        _verify_and_save(builder, filename, data, is_td=True)


def _verify_and_save(builder: VRPInstanceBuilder, filename: str, data: DataConfig, is_td: bool = False) -> None:
    """Verify file existence and save the dataset."""
    ext = ".td" if is_td else ".npz"
    assert data.overwrite or not os.path.isfile(check_extension(filename, ext)), (
        "File already exists! Try running with -f option to overwrite."
    )

    if is_td:
        dataset = builder.build_td()
        save_td_dataset(dataset, filename)
    else:
        dataset = builder.build()
        save_simulation_dataset(dataset, filename)
