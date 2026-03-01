"""datasets.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import datasets
"""

import os
import random
import time
from typing import Any, Optional

import numpy as np
import torch

from logic.src.configs import Config
from logic.src.configs.envs.graph import GraphConfig
from logic.src.configs.tasks.data import DataConfig
from logic.src.constants import ROOT_DIR
from logic.src.data.generators.builders import VRPInstanceBuilder
from logic.src.pipeline.simulations.repository import set_repository_from_path
from logic.src.tracking.logging.pylogger import get_pylogger
from logic.src.utils.data.loader import (
    check_extension,
    save_simulation_dataset,
    save_td_dataset,
)

logger = get_pylogger(__name__)


def generate_datasets(cfg: Config) -> None:
    """
    Generates VRP datasets based on the typed Hydra configuration.

    Args:
        cfg: Root Hydra configuration with ``cfg.data`` containing data
            generation parameters.
    """
    from logic.src.data.generators.validators import validate_data_config
    from logic.src.tracking.core.run import get_active_run

    t0 = time.perf_counter()

    # Validate and sanitize config values
    validate_data_config(cfg)

    # Initialize the filesystem repository for coordinate/depot loading
    set_repository_from_path(str(ROOT_DIR))

    data = cfg.data

    # --- Log config params to WSTracker ---
    run = get_active_run()
    if run is not None:
        run.log_params(
            {
                "gen/problem": data.problem,
                "gen/dataset_type": data.dataset_type,
                "gen/seed": data.seed,
                "gen/n_epochs": data.n_epochs,
                "gen/overwrite": data.overwrite,
            }
        )
        run.set_tag("task", "gen_data")

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

    files_generated = 0
    for problem, distributions in problems.items():
        files_generated += _generate_problem_data(problem, distributions, data)

    # --- Log summary metrics ---
    elapsed = time.perf_counter() - t0
    if run is not None:
        run.log_metric("gen/total_elapsed_s", round(elapsed, 3), step=0)
        run.log_metric("gen/num_files_generated", files_generated, step=0)
    logger.info(f"Data generation complete: {files_generated} files in {elapsed:.1f}s")


def _generate_problem_data(problem: str, distributions: Any, data: DataConfig) -> int:
    """Helper to generate data for a specific problem and its distributions.

    Returns:
        Number of files generated.
    """
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
    file_count = 0

    try:
        for dist in distributions or [None]:
            if graphs:
                for graph_cfg in graphs:
                    file_count += _process_instance_generation(problem, dist, datadir, data, graph_cfg=graph_cfg)
            else:
                print("[WARNING] No graphs provided for instance generation. Skipping.")
    except Exception as e:
        has_dists = len(distributions) >= 1 and distributions[0] is not None
        raise Exception(
            "failed to generate data for problem {}{}  due to {}".format(
                problem, f" {distributions}" if has_dists else "", repr(e)
            )
        ) from e

    return file_count


def _process_instance_generation(
    problem: str,
    dist: Any,
    datadir: str,
    data: DataConfig,
    graph_cfg: Optional[GraphConfig] = None,
) -> int:
    """Configure builder and save datasets for a specific configuration.

    Returns:
        Number of files generated.
    """
    from logic.src.tracking.core.run import get_active_run

    assert graph_cfg is not None, "graph_cfg must be provided"
    size = graph_cfg.num_loc
    graph = graph_cfg.focus_graph
    area = graph_cfg.area
    waste_type = graph_cfg.waste_type
    vertex_method = graph_cfg.vertex_method
    focus_size = graph_cfg.focus_size if graph_cfg.focus_size is not None else 0
    n_days = graph_cfg.n_days if graph_cfg.n_days is not None else max(data.n_epochs - data.epoch_start, 1)
    n_samples = graph_cfg.n_samples if graph_cfg.n_samples is not None else 1
    if data.dataset_type == "train":
        n_days = 1

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
    config_label = f"{problem}_{dist or 'none'}_{size}"
    logger.info(
        "Generating '{}{}' ({}) dataset for {} with {} locations [dist={}]".format(
            name,
            n_days if data.dataset_type != "train" else "",
            data.dataset_type or "train",
            problem.upper(),
            size,
            dist,
        )
    )

    # --- Log per-config params ---
    run = get_active_run()
    if run is not None:
        run.log_params(
            {
                f"gen/config/{config_label}/area": area,
                f"gen/config/{config_label}/waste_type": waste_type,
                f"gen/config/{config_label}/n_days": n_days,
                f"gen/config/{config_label}/n_samples": n_samples,
                f"gen/config/{config_label}/dataset_type": data.dataset_type or "train",
            }
        )

    t0 = time.perf_counter()

    builder = VRPInstanceBuilder()
    builder.set_dataset_size(n_samples).set_problem_size(size).set_waste_type(waste_type).set_distribution(
        dist or "empty"
    ).set_area(area).set_focus_graph(graph, focus_size).set_method(vertex_method).set_problem_name(
        problem
    ).set_num_days(n_days)

    _apply_noise_config(builder, problem, data)

    file_count = 0
    if data.dataset_type == "test_simulator":
        file_count = _generate_test_simulator_data(
            builder, n_days, datadir, dist, size, data, area, waste_type, n_samples
        )
    elif data.dataset_type == "train_time":
        file_count = _generate_train_time_data(builder, problem, n_days, datadir, dist, size, data)
    else:
        file_count = _generate_train_data(builder, problem, datadir, dist, size, data)

    elapsed = time.perf_counter() - t0
    if run is not None:
        run.log_metric(f"gen/{config_label}/elapsed_s", round(elapsed, 3), step=0)
    logger.debug(f"Config {config_label} generated {file_count} file(s) in {elapsed:.2f}s")
    return file_count


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
) -> int:
    """Generate and save test simulator data.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
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
    return 1


def _generate_train_time_data(
    builder: VRPInstanceBuilder, problem: str, n_days: int, datadir: str, dist: Any, size: int, data: DataConfig
) -> int:
    """Generate and save train time data.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
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
    return 1


def _generate_train_data(
    builder: VRPInstanceBuilder, problem: str, datadir: str, dist: Any, size: int, data: DataConfig
) -> int:
    """Generate and save standard training data.

    Returns:
        Number of files generated.
    """
    name = data.name or "dataset"
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
    return max(data.n_epochs - data.epoch_start, 0)


def _verify_and_save(builder: VRPInstanceBuilder, filename: str, data: DataConfig, is_td: bool = False) -> None:
    """Verify file existence and save the dataset."""
    import contextlib

    from logic.src.tracking.core.run import get_active_run
    from logic.src.tracking.integrations.data import RuntimeDataTracker

    ext = ".td" if is_td else ".npz"
    assert data.overwrite or not os.path.isfile(check_extension(filename, ext)), (
        f"File {filename} already exists! Try running with -f option to overwrite."
    )

    if is_td:
        dataset = builder.build_td()
        save_td_dataset(dataset, filename)
    else:
        dataset = builder.build()
        save_simulation_dataset(dataset, filename)

    # --- WSTracker: log artifact and dataset statistics ---
    run = get_active_run()
    if run is not None:
        saved_path = check_extension(filename, ext)

        # Log file as artifact
        with contextlib.suppress(Exception):
            file_size = os.path.getsize(saved_path) if os.path.exists(saved_path) else 0
            run.log_artifact(
                saved_path,
                artifact_type="dataset",
                metadata={"format": ext.lstrip("."), "file_size_bytes": file_size},
            )
            run.log_metric("gen/file_size_bytes", file_size, step=0)

        # Snapshot tensor statistics via RuntimeDataTracker
        with contextlib.suppress(Exception):
            tracker = RuntimeDataTracker(run)
            tracker.on_load(
                dataset,
                metadata={
                    "filename": os.path.basename(saved_path),
                    "variable_name": "dataset",
                    "source_file": "generators/datasets.py",
                    "source_line": 428,
                },
            )
