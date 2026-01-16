"""
Simulation and evaluation related argument parsers.
"""
import os
import argparse
import re

import re
from multiprocessing import cpu_count
from logic.src.cli.base_parser import LowercaseAction, StoreDictKeyPair
from logic.src.utils.functions import parse_softmax_temperature
from logic.src.utils.definitions import (
    MAP_DEPOTS, WASTE_TYPES
)

def add_eval_args(parser):
    """
    Adds all arguments related to evaluation to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added evaluation arguments.
    """
    parser.add_argument(
        "--datasets", type=str, nargs="+", help="Filename of the dataset(s) to evaluate"
    )
    parser.add_argument(
        "-f", action="store_true", dest="overwrite", help="Set true to overwrite"
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        default=None,
        help="Name of the results file to write",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=12_800,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset where to start in dataset"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decode type, greedy or sampling",
    )
    parser.add_argument(
        "--width",
        type=int,
        nargs="+",
        help="Sizes of beam to use for beam search (or number of samples for sampling), "
        "0 to disable (default), -1 for infinite",
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        help="Beam search (bs), Sampling (sample) or Greedy (greedy)",
    )
    parser.add_argument(
        "--softmax_temperature",
        type=parse_softmax_temperature,
        default=1,
        help="Softmax temperature (sampling or bs)",
    )
    parser.add_argument("--model", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    parser.add_argument(
        "--data_distribution",
        type=str,
        default=None,
        help="Data distribution of the dataset",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--compress_mask", action="store_true", help="Compress mask into long"
    )
    parser.add_argument(
        "--max_calc_batch_size", type=int, default=12_800, help="Size for subbatches"
    )
    parser.add_argument(
        "--results_dir", default="results", help="Name of evaluation results directory"
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Use multiprocessing to parallelize over multiple GPUs",
    )
    parser.add_argument(
        "--graph_size", type=int, default=50, help="The size of the problem graph"
    )
    parser.add_argument(
        "--area", type=str, default="riomaior", help="County area of the bins locations"
    )
    parser.add_argument(
        "--waste_type",
        type=str,
        default="plastic",
        help="Type of waste bins selected for the optimization problem",
    )
    parser.add_argument(
        "--focus_graph",
        default=None,
        help="Path to the file with the coordinates of the graph to focus on",
    )
    parser.add_argument(
        "--focus_size",
        type=int,
        default=0,
        help="Number of focus graphs to include in the training data",
    )
    parser.add_argument(
        "--edge_threshold",
        default="0",
        type=str,
        help="How many of all possible edges to consider",
    )
    parser.add_argument(
        "--edge_method",
        type=str,
        default=None,
        help="Method for getting edges ('dist'|'knn')",
    )
    parser.add_argument(
        "--distance_method",
        type=str,
        default="ogd",
        help="Method to compute distance matrix",
    )
    parser.add_argument(
        "--dm_filepath", type=str, default=None, help="Path to the distance matrix file"
    )
    parser.add_argument(
        "--vertex_method",
        type=str,
        default="mmn",
        help="Method to transform vertex coordinates "
        "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'",
    )
    parser.add_argument(
        "--w_length", type=float, default=1.0, help="Weight for length in cost function"
    )
    parser.add_argument(
        "--w_waste", type=float, default=1.0, help="Weight for waste in cost function"
    )
    parser.add_argument(
        "--w_overflows",
        type=float,
        default=1.0,
        help="Weight for overflows in cost function",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="cwcvrp",
        help="Problem to evaluate ('wcvrp'|'cwcvrp'|'sdwcvrp'|'scwcvrp')",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="gat",
        help="Encoder to use ('gat'|'gac'|'tgc'|'ggac')",
    )
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state from"
    )
    return parser

def add_test_sim_args(parser):
    """
    Adds all arguments related to the test simulator to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added test simulator arguments.
    """
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        required=True,
        help="Name of the policy(ies) to test on the WSR simulator",
    )
    parser.add_argument(
        "--gate_prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for gating decisions (default: 0.5)",
    )
    parser.add_argument(
        "--mask_prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for mask decisions (default: 0.5)",
    )
    parser.add_argument(
        "--data_distribution",
        "--dd",
        type=str,
        default="gamma1",
        help="Distribution to generate the bins daily waste fill",
    )
    parser.add_argument(
        "--problem", default="vrpp", help="The problem the model was trained to solve"
    )
    parser.add_argument(
        "--size", type=int, default=50, help="The size of the problem graph"
    )
    parser.add_argument(
        "--days", type=int, default=31, help="Number of days to run the simulation for"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Name of WSR simulator test output directory",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="temp",
        help="Name of WSR simulator test runs checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint_days",
        type=int,
        default=0,
        help="Number of days interval to save simulation checkpoints (0 to deactivate checkpointing)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the system logger",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/simulation.log",
        help="Path to the system log file",
    )
    parser.add_argument(
        "--cpd",
        type=int,
        default=5,
        help="Save checkpoint every n days, 0 to save no checkpoints",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of simulation samplings for each policy",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume testing (relevant for saving results)",
    )
    parser.add_argument(
        "--pregular_level", "--lvl", type=int, nargs="+", help="Regular policy level"
    )
    parser.add_argument(
        "--plastminute_cf",
        "--cf",
        type=int,
        nargs="+",
        help="CF value for last minute/last minute and path policies",
    )
    parser.add_argument(
        "--lookahead_configs",
        "--lac",
        type=str,
        nargs="+",
        help="Parameter configuration for policy Look-Ahead and variants",
    )
    parser.add_argument(
        "--gurobi_param",
        "--gp",
        type=float,
        default=0.84,
        nargs="+",
        help="Param value for Gurobi VRPP policy "
        "(higher = more conservative with regards to amount of overflows)",
    )
    parser.add_argument(
        "--hexaly_param",
        "--hp",
        type=float,
        default=2.0,
        nargs="+",
        help="Param value for Hexaly optimizer policy "
        "(higher = more conservative with regards to amount of overflows)",
    )
    parser.add_argument(
        "--cpu_cores",
        "--cc",
        type=int,
        default=0,
        help="Number of max CPU cores to use (0 uses all available cores)",
    )
    parser.add_argument("--n_vehicles", type=int, default=1, help="Number of vehicles")
    parser.add_argument(
        "--area", type=str, default="riomaior", help="County area of the bins locations"
    )
    parser.add_argument(
        "--waste_type",
        type=str,
        default="plastic",
        help="Type of waste bins selected for the optimization problem",
    )
    parser.add_argument(
        "--bin_idx_file",
        type=str,
        default=None,
        help="File with the indices of the bins to use in the simulation",
    )
    parser.add_argument(
        "--decode_type",
        "--dt",
        type=str,
        default="greedy",
        help="Decode type, greedy or sampling",
    )
    parser.add_argument(
        "--temperature",
        type=parse_softmax_temperature,
        default=1,
        help="Softmax temperature (sampling or bs)",
    )
    parser.add_argument(
        "--edge_threshold",
        "--et",
        default="0",
        type=str,
        help="How many of all possible edges to consider",
    )
    parser.add_argument(
        "--edge_method",
        "--em",
        type=str,
        default=None,
        help="Method for getting edges ('dist'|'knn')",
    )
    parser.add_argument(
        "--vertex_method",
        "--vm",
        type=str,
        default="mmn",
        help="Method to transform vertex coordinates "
        "'mmn'|'mun'|'smsd'|'ecp'|'utmp'|'wmp'|'hdp'|'c3d'|'s4d'",
    )
    parser.add_argument(
        "--distance_method",
        "--dm",
        type=str,
        default="ogd",
        help="Method to compute distance matrix",
    )
    parser.add_argument(
        "--dm_filepath",
        "--dmf",
        type=str,
        default=None,
        help="Path to the file to read/write the distance matrix from/to",
    )
    parser.add_argument(
        "--waste_filepath",
        type=str,
        default=None,
        help="Path to the file to read the waste fill for each day from",
    )
    parser.add_argument(
        "--noise_mean",
        type=float,
        default=0.0,
        help="Mean of Gaussian noise to inject into observed bin levels",
    )
    parser.add_argument(
        "--noise_variance",
        type=float,
        default=0.0,
        help="Variance of Gaussian noise to inject into observed bin levels",
    )
    parser.add_argument(
        "--run_tsp", action="store_true", help="Activate fast_tsp for all policies."
    )
    parser.add_argument(
        "--two_opt_max_iter",
        type=int,
        default=0,
        help="Maximum number of 2-opt iterations",
    )
    parser.add_argument(
        "--cache_regular",
        action="store_false",
        help="Deactivate caching for policy regular.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--server_run",
        action="store_true",
        help="Simulation will be executed in a remote server",
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default="vars.env",
        help="Name of the file that contains the environment variables",
    )
    parser.add_argument(
        "--gplic_file",
        type=str,
        default=None,
        help="Name of the file that contains the license to use for Gurobi",
    )
    parser.add_argument(
        "--hexlic_file",
        type=str,
        default=None,
        help="Name of the file that contains the license to use for Gurobi",
    )
    parser.add_argument(
        "--symkey_name",
        type=str,
        default=None,
        help="Name of the cryptographic key used to access the API keys",
    )
    parser.add_argument(
        "--gapik_file",
        type=str,
        default=None,
        help="Name of the file that contains the key to use for the Google API",
    )
    parser.add_argument(
        "--real_time_log", action="store_true", help="Activate real time results window"
    )
    parser.add_argument(
        "--stats_filepath",
        type=str,
        default=None,
        help="Path to the file to read the statistics from",
    )
    parser.add_argument(
        "--model_path",
        action=StoreDictKeyPair,
        default=None,
        nargs="+",
        help="Path to the directory where the model(s) is/are stored (format: name=path)",
    )
    parser.add_argument(
        "--config_path",
        action=StoreDictKeyPair,
        default=None,
        nargs="+",
        help="Path to the YAML/XML configuration file(s) (format: name=path)",
    )
    return parser

def validate_test_sim_args(args):
    """
    Validates and post-processes arguments for test_sim.
    """
    args = args.copy()
    assert args.get("days", 0) >= 1, "Must run the simulation for 1 or more days"
    assert (
        args.get("n_samples", 0) > 0
    ), "Number of samples must be non-negative integer"

    args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
    assert (
        args["area"] in MAP_DEPOTS.keys()
    ), "Unknown area {}, available areas: {}".format(args["area"], MAP_DEPOTS.keys())

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert (
        args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None
    ), "Unknown waste type {}, available waste types: {}".format(
        args["waste_type"], WASTE_TYPES.keys()
    )

    args["edge_threshold"] = (
        float(args["edge_threshold"])
        if "." in str(args.get("edge_threshold", "0"))
        else int(args.get("edge_threshold", "0"))
    )

    assert (
        args.get("cpu_cores", 0) >= 0
    ), "Number of CPU cores must be non-negative integer"
    assert (
        args.get("cpu_cores", 0) <= cpu_count()
    ), "Number of CPU cores to use cannot exceed system specifications"
    if args.get("cpu_cores") == 0:
        args["cpu_cores"] = cpu_count()

    if args.get("plastminute_cf"):
        vals = (
            args["plastminute_cf"]
            if isinstance(args["plastminute_cf"], list)
            else [args["plastminute_cf"]]
        )
        for cf in vals:
            assert (
                cf > 0 and cf < 100
            ), "Policy last minute CF must be between 0 and 100"
    if args.get("pregular_level"):
        vals = (
            args["pregular_level"]
            if isinstance(args["pregular_level"], list)
            else [args["pregular_level"]]
        )
        for lvl in vals:
            assert (
                lvl >= 1 and lvl <= args["days"]
            ), "Policy regular level must be between 1 and number of days, inclusive"
    if args.get("gurobi_param"):
        vals = (
            args["gurobi_param"]
            if isinstance(args["gurobi_param"], list)
            else [args["gurobi_param"]]
        )
        for gp in vals:
            assert gp > 0, "Policy gurobi parameter must be greater than 0"
    if args.get("hexaly_param"):
        vals = (
            args["hexaly_param"]
            if isinstance(args["hexaly_param"], list)
            else [args["hexaly_param"]]
        )
        for hp in vals:
            assert hp > 0, "Policy hexaly parameter must be greater than 0"
    if args.get("lookahead_configs"):
        vals = (
            args["lookahead_configs"]
            if isinstance(args["lookahead_configs"], list)
            else [args["lookahead_configs"]]
        )
        for lac in vals:
            assert lac in [
                "a",
                "b",
            ], "Policy lookahead configuration must be 'a' or 'b'"
    return args


def validate_eval_args(args):
    """
    Validates and post-processes arguments for eval.
    """
    args = args.copy()
    # Handle the -o alias for output_filename
    if "output_filename" in args and args["output_filename"] is not None:
        args["o"] = args["output_filename"]

    assert (
        "o" not in args
        or args["o"] is None
        or (len(args.get("datasets") or []) == 1 and len(args.get("width") or []) <= 1)
    ), "Cannot specify result filename with more than one dataset or more than one width"

    args["area"] = re.sub(r"[^a-zA-Z]", "", args.get("area", "").lower())
    assert (
        args["area"] in MAP_DEPOTS.keys()
    ), "Unknown area {}, available areas: {}".format(args["area"], MAP_DEPOTS.keys())

    args["waste_type"] = re.sub(r"[^a-zA-Z]", "", args.get("waste_type", "").lower())
    assert (
        args["waste_type"] in WASTE_TYPES.keys() or args["waste_type"] is None
    ), "Unknown waste type {}, available waste types: {}".format(
        args["waste_type"], WASTE_TYPES.keys()
    )
    return args
