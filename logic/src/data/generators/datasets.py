"""datasets.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import datasets
"""

import os
import random
from typing import Any, Dict

import numpy as np
import torch

from logic.src.constants import ROOT_DIR
from logic.src.data.builders import VRPInstanceBuilder
from logic.src.utils.data.data_utils import check_extension, save_dataset, save_td_dataset


def generate_datasets(opts: Dict[str, Any]) -> None:
    """
    Generates VRP datasets based on the provided configuration options.
    """
    # Set the random seed and execute the program
    random.seed(opts["seed"])
    np.random.seed(opts["seed"])
    torch.manual_seed(opts["seed"])

    gamma_dists = ["gamma1", "gamma2", "gamma3", "gamma4"]
    distributions_per_problem = {
        "vrpp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "wcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
        "swcvrp": ["empty", "const", "unif", "dist", "emp", *gamma_dists],
    }

    # Define the problem distribution(s)
    if opts["problem"] == "all":
        problems = distributions_per_problem
    else:
        problems = {
            opts["problem"]: (
                distributions_per_problem[opts["problem"]]
                if len(opts["data_distributions"]) == 1 and opts["data_distributions"][0] == "all"
                else [data_dist for data_dist in opts["data_distributions"]]
            )
        }

    n_days = opts["n_epochs"] - opts["epoch_start"] if opts["dataset_type"] != "train" else 0
    for problem, distributions in problems.items():
        _generate_problem_data(problem, distributions, n_days, opts)


def _generate_problem_data(problem: str, distributions: Any, n_days: int, opts: Dict[str, Any]) -> None:
    """Helper to generate data for a specific problem and its distributions."""
    datadir = (
        os.path.join(ROOT_DIR, "data", opts["data_dir"], problem)
        if opts["dataset_type"] in ["train_time", "train"]
        else os.path.join(ROOT_DIR, "data", "wsr_simulator", opts["data_dir"])
    )
    try:
        os.makedirs(datadir, exist_ok=True)
    except Exception as e:
        raise Exception("directories to save generated data files do not exist and could not be created") from e

    try:
        for dist in distributions or [None]:
            for size, graph in zip(opts["num_locs"], opts["focus_graph"]):
                _process_instance_generation(problem, dist, size, graph, n_days, datadir, opts)
    except Exception as e:
        has_dists = len(distributions) >= 1 and distributions[0] is not None
        raise Exception(
            "failed to generate data for problem {}{} due to {}".format(
                problem, f" {distributions}" if has_dists else "", repr(e)
            )
        ) from e


def _process_instance_generation(
    problem: str, dist: Any, size: int, graph: Any, n_days: int, datadir: str, opts: Dict[str, Any]
) -> None:
    """Configure builder and save datasets for a specific configuration."""
    if graph is not None and not os.path.isfile(graph):
        sim_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator")
        if os.path.isfile(os.path.join(sim_dir, graph)):
            graph = os.path.join(sim_dir, graph)
        elif os.path.isfile(os.path.join(sim_dir, "bins_selection", graph)):
            graph = os.path.join(sim_dir, "bins_selection", graph)

    print(
        "Generating '{}{}' ({}) dataset for the {} with {} locations{}{}".format(
            opts["name"],
            n_days if n_days > 0 else "",
            opts["dataset_type"],
            problem.upper(),
            size,
            " and using '{}' as the instance distribution".format(dist),
            ":" if n_days == 0 else "...",
        )
    )

    builder = VRPInstanceBuilder()
    builder.set_dataset_size(opts["dataset_size"]).set_problem_size(size).set_waste_type(
        opts["waste_type"]
    ).set_distribution(dist or "empty").set_area(opts["area"]).set_focus_graph(graph, opts["focus_size"]).set_method(
        opts["vertex_method"]
    ).set_problem_name(problem)

    _apply_noise_config(builder, problem, opts)

    if opts["dataset_type"] == "test_simulator":
        _generate_test_simulator_data(builder, n_days, datadir, dist, size, opts)
    elif opts["dataset_type"] == "train_time":
        _generate_train_time_data(builder, problem, n_days, datadir, dist, size, opts)
    else:
        _generate_train_data(builder, problem, datadir, dist, size, opts)


def _apply_noise_config(builder: VRPInstanceBuilder, problem: str, opts: Dict[str, Any]) -> None:
    """Apply noise parameters to the builder."""
    if opts.get("mu") is not None:
        sigma = opts.get("sigma", 1.0)
        if isinstance(sigma, list):
            sigma = sigma[0]
        builder.set_noise(opts["mu"], (sigma**2 if sigma > 0 else 0.0))
    elif problem == "swcvrp":
        sigma = opts.get("sigma", 1.0)
        if isinstance(sigma, list):
            sigma = sigma[0]
        mu = opts.get("mu", 0.0)
        builder.set_noise(
            mu if mu is not None else 0.0,
            (sigma**2 if sigma > 0 else 1.0),
        )


def _generate_test_simulator_data(
    builder: VRPInstanceBuilder, n_days: int, datadir: str, dist: Any, size: int, opts: Dict[str, Any]
) -> None:
    """Generate and save test simulator data."""
    builder.set_num_days(n_days)
    if "filename" not in opts or opts["filename"] is None:
        filename = os.path.join(
            datadir,
            "{}{}{}_{}{}_N{}_seed{}.pkl".format(
                opts["area"],
                size,
                "_{}".format(dist) if dist is not None else "",
                opts["name"],
                n_days if n_days > 1 else "",
                opts["dataset_size"],
                opts["seed"],
            ),
        )
    else:
        filename = check_extension(opts["filename"])

    _verify_and_save(builder, filename, opts, is_td=False)


def _generate_train_time_data(
    builder: VRPInstanceBuilder, problem: str, n_days: int, datadir: str, dist: Any, size: int, opts: Dict[str, Any]
) -> None:
    """Generate and save train time data."""
    builder.set_num_days(opts["n_epochs"])
    if "filename" not in opts or opts["filename"] is None:
        ext = ".td"
        if opts.get("mu") is not None:
            filename = os.path.join(
                datadir,
                "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                    problem,
                    size,
                    "_{}".format(dist) if dist is not None else "",
                    opts["name"],
                    n_days if n_days > 1 else "",
                    opts["seed"],
                    f"gaussian{opts['mu']}",
                    f"_{opts['sigma']}",
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
                    opts["name"],
                    n_days if n_days > 1 else "",
                    opts["seed"],
                    ext,
                ),
            )
    else:
        filename = check_extension(opts["filename"], ".td")

    _verify_and_save(builder, filename, opts, is_td=True)


def _generate_train_data(
    builder: VRPInstanceBuilder, problem: str, datadir: str, dist: Any, size: int, opts: Dict[str, Any]
) -> None:
    """Generate and save standard training data."""
    builder.set_num_days(1)
    for epoch in range(opts["epoch_start"], opts["n_epochs"]):
        print("- Generating epoch {} data".format(epoch))
        if "filename" not in opts or opts["filename"] is None:
            ext = ".td"
            if opts.get("mu") is not None:
                filename = os.path.join(
                    datadir,
                    "{}{}{}_{}{}_seed{}_{}_{}{}".format(
                        problem,
                        size,
                        ("_{}".format(dist) if dist is not None else ""),
                        opts["name"],
                        epoch if opts["n_epochs"] > 1 else "",
                        opts["seed"],
                        f"gaussian{opts['mu']}",
                        f"_{opts['sigma']}",
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
                        opts["name"],
                        epoch if opts["n_epochs"] > 1 else "",
                        opts["seed"],
                        ext,
                    ),
                )
        else:
            filename = check_extension(opts["filename"], ".td")

        _verify_and_save(builder, filename, opts, is_td=True)


def _verify_and_save(builder: VRPInstanceBuilder, filename: str, opts: Dict[str, Any], is_td: bool = False) -> None:
    """Verify file existence and save the dataset."""
    ext = ".td" if is_td else ".pkl"
    assert opts.get("f", opts.get("overwrite", False)) or not os.path.isfile(check_extension(filename, ext)), (
        "File already exists! Try running with -f option to overwrite."
    )

    if is_td:
        dataset = builder.build_td()
        save_td_dataset(dataset, filename)
    else:
        full_dataset = builder.build()
        dataset = [(x[2], x[3]) for x in full_dataset] if len(full_dataset[0]) == 5 else [x[2] for x in full_dataset]
        save_dataset(dataset, filename)
