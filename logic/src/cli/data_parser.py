"""
Data generation related argument parsers.
"""

import argparse
import re
from typing import Any, Dict

from logic.src.utils.definitions import MAP_DEPOTS, WASTE_TYPES


def add_gen_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all arguments related to data generation to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser instance with data generation arguments attached.
    """
    parser.add_argument("--name", type=str, help="Name to identify dataset")
    parser.add_argument(
        "--filename",
        default=None,
        help="Filename of the dataset to create (ignores datadir)",
    )
    parser.add_argument("--data_dir", default="datasets", help="Create datasets in data")
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem type selection. Should be 'vrpp', 'wcvrp', 'swcvrp', or 'all'.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        nargs="+",
        help="Mean of Gaussian noise (implies Gaussian noise generation if set)",
    )
    parser.add_argument("--sigma", type=float, nargs="+", default=0.6, help="Variance of Gaussian noise")
    parser.add_argument(
        "--data_distributions",
        nargs="+",
        default=["all"],
        help="Distributions to generate for problems",
    )
    parser.add_argument("--dataset_size", type=int, default=128_000, help="Size of the dataset")
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Sizes of problem instances",
    )
    parser.add_argument("--penalty_factor", type=float, default=3.0, help="Penalty factor for problems")
    parser.add_argument("-f", action="store_true", dest="overwrite", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="The number of epochs to generate data for",
    )
    parser.add_argument("--epoch_start", type=int, default=0, help="Start at epoch #")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["train", "train_time", "test_simulator"],
        help="Set type of dataset to generate",
    )
    parser.add_argument("--area", type=str, default="riomaior", help="County area of the bins locations")
    parser.add_argument(
        "--waste_type",
        type=str,
        default="plastic",
        help="Type of waste bins selected for the optimization problem",
    )
    parser.add_argument(
        "--focus_graphs",
        nargs="+",
        default=None,
        help="Path to the files with the coordinates of the graphs to focus on",
    )
    parser.add_argument(
        "--focus_size",
        type=int,
        default=0,
        help="Number of focus graphs to include in the data",
    )
    parser.add_argument(
        "--vertex_method",
        type=str,
        default="mmn",
        help="Method to transform vertex coordinates 'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'",
    )
    return parser


def validate_gen_data_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and post-process arguments for data generation.

    This function checks consistency between arguments, such as ensuring a
    filename is only provided when a single dataset is being generated.

    Args:
        args: Dictionary of parsed arguments.

    Returns:
        The validated and potentially modified dictionary of arguments.

    Raises:
        AssertionError: If incompatible arguments are provided (e.g., mu/sigma length mismatch).
    """
    args = args.copy()
    assert (
        "filename" not in args
        or args["filename"] is None
        or (
            (isinstance(args.get("problem"), str) and args.get("problem") != "all")
            or (isinstance(args.get("problem"), list) and len(args["problem"]) == 1)
        )
        and len(args.get("graph_sizes", [])) == 1
    ), "Can only specify filename when generating a single dataset"

    if args["problem"] in ["all", "swcvrp"]:
        assert "mu" in args and args["mu"] is not None, "Must specify mu when generating swcvrp datasets"
        assert "sigma" in args and args["sigma"] is not None, "Must specify sigma when generating swcvrp datasets"
        assert len(args["mu"]) == len(args["sigma"]), "Must specify same number of mu and sigma values"

    assert (
        "focus_graphs" not in args
        or args["focus_graphs"] is None
        or len(args["focus_graphs"]) == len(args.get("graph_sizes", []))
    )

    if "focus_graphs" not in args or args["focus_graphs"] is None:
        args["focus_graphs"] = [None] * len(args.get("graph_sizes", []))
    else:
        args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
        assert args["area"] in MAP_DEPOTS.keys(), "Unknown area {}, available areas: {}".format(
            args["area"], MAP_DEPOTS.keys()
        )

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert (
        args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None
    ), "Unknown waste type {}, available waste types: {}".format(args["waste_type"], WASTE_TYPES.keys())
    return args
