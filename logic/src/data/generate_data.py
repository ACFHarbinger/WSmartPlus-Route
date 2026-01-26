"""
Data generation scripts for VRP/VRPP.

This module provides utilities to generate synthetic datasets for Value Routing Problems (VRP)
and their variants (VRPP, WCVRP) with configurable parameters.
"""

import argparse
import os
import random
import sys
import traceback

import numpy as np
import torch

from logic.src.cli.base_parser import ConfigsParser
from logic.src.cli.data_parser import add_gen_data_args, validate_gen_data_args
from logic.src.constants import ROOT_DIR
from logic.src.data.builders import VRPInstanceBuilder
from logic.src.utils.data.data_utils import check_extension, save_dataset, save_td_dataset


def generate_datasets(opts):
    """
    Generates VRP datasets based on the provided configuration options.

    Args:
        opts (dict): A dictionary containing configuration parameters for data generation.
                     Expected keys include 'seed', 'problem', 'dataset_type', 'graph_sizes',
                     'waste_type', 'area', 'vertex_method', 'dataset_size', etc.

    Raises:
        Exception: If directory creation fails or data generation encounters an error.
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

    # Generate the dataset(s)
    n_days = opts["n_epochs"] - opts["epoch_start"] if opts["dataset_type"] != "train" else 0
    for problem, distributions in problems.items():
        datadir = (
            os.path.join(ROOT_DIR, "data", opts["data_dir"], problem)
            if opts["dataset_type"] in ["train_time", "train"]
            else os.path.join(ROOT_DIR, "data", "wsr_simulator", opts["data_dir"])
        )
        try:
            os.makedirs(datadir, exist_ok=True)  # Create directory for saving data
        except Exception:
            raise Exception("directories to save generated data files do not exist and could not be created")

        try:
            for dist in distributions or [None]:
                for size, graph in zip(opts["graph_sizes"], opts["focus_graphs"]):
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

                    # Configure Builder
                    builder = VRPInstanceBuilder()
                    builder.set_dataset_size(opts["dataset_size"]).set_problem_size(size).set_waste_type(
                        opts["waste_type"]
                    ).set_distribution(dist).set_area(opts["area"]).set_focus_graph(
                        graph, opts["focus_size"]
                    ).set_method(opts["vertex_method"]).set_problem_name(problem)

                    if opts.get("mu") is not None:
                        sigma = opts.get("sigma", 1.0)
                        if isinstance(sigma, list):
                            sigma = sigma[0]
                        builder.set_noise(opts["mu"], sigma**2 if sigma > 0 else 0.0)
                    elif problem == "swcvrp":
                        sigma = opts.get("sigma", 1.0)
                        if isinstance(sigma, list):
                            sigma = sigma[0]
                        mu = opts.get("mu", 0.0)
                        builder.set_noise(
                            mu if mu is not None else 0.0,
                            sigma**2 if sigma > 0 else 1.0,
                        )

                    if opts["dataset_type"] == "test_simulator":
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

                        assert opts.get("f", opts.get("overwrite", False)) or not os.path.isfile(
                            check_extension(filename)
                        ), "File already exists! Try running with -f option to overwrite."

                        full_dataset = builder.build()
                        # Verify what generate_wsr_data returned. It returned just the waste list.
                        # VRPInstanceBuilder returns list of (depot, loc, waste, max_waste).
                        # We extract waste (index 2).
                        if len(full_dataset[0]) == 5:
                            dataset = [(x[2], x[3]) for x in full_dataset]
                        else:
                            dataset = [x[2] for x in full_dataset]

                        save_dataset(dataset, filename)

                    elif opts["dataset_type"] == "train_time":
                        builder.set_num_days(opts["n_epochs"])

                        if "filename" not in opts or opts["filename"] is None:
                            ext = ".td"
                            if opts.get("mu", None) is not None:
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

                        assert opts.get("f", opts.get("overwrite", False)) or not os.path.isfile(
                            check_extension(filename, ".td")
                        ), "File already exists! Try running with -f option to overwrite."

                        dataset = builder.build_td()
                        save_td_dataset(dataset, filename)
                    else:
                        assert opts["dataset_type"] == "train"
                        builder.set_num_days(1)

                        for epoch in range(opts["epoch_start"], opts["n_epochs"]):
                            print("- Generating epoch {} data".format(epoch))
                            if "filename" not in opts or opts["filename"] is None:
                                ext = ".td"
                                if opts.get("mu", None) is not None:
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

                            assert opts.get("f", opts.get("overwrite", False)) or not os.path.isfile(
                                check_extension(filename, ".td")
                            ), "File already exists! Try running with -f option to overwrite."

                            dataset = builder.build_td()
                            save_td_dataset(dataset, filename)
        except Exception as e:
            has_dists = len(distributions) >= 1 and distributions[0] is not None
            raise Exception(
                "failed to generate data for problem {}{} due to {}".format(
                    problem, f" {distributions}" if has_dists else "", repr(e)
                )
            )


if __name__ == "__main__":
    exit_code = 0
    parser = ConfigsParser(
        description="Data Generator Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    add_gen_data_args(parser)
    try:
        _, parsed_args_dict = parser.parse_process_args(sys.argv[1:], "gen_data")
        args = validate_gen_data_args(parsed_args_dict)
        generate_datasets(args)
    except (argparse.ArgumentError, AssertionError) as e:
        exit_code = 1
        parser.print_help()
        print(f"Error: {e}", file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(str(e), file=sys.stderr)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
