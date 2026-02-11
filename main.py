#!/usr/bin/env python
"""
Main Entry Point for the WSmart-Route Application.

This module serves as the primary gateway for executing various components of the
project, including the WSmart-Route simulator, reinforcement learning training
pipelines, test suite execution, and the Graphical User Interface (GUI).

It dispatches commands based on arguments parsed by `logic.src.cli`
to the appropriate subsystems (Logic, GUI, Test).

Key Functions:
    - run_test_suite: Orchestrates the execution of unit and integration tests.
    - pretty_print_args: Formats and displays command-line arguments for logging.
    - main: The central dispatcher that routes the parsed CLI arguments to the
      corresponding execution logic.
"""

import io
import pprint
import sys
import traceback
import warnings
from typing import Any, cast

import hydra
from gui.src.app import run_app_gui
from hydra.core.config_store import ConfigStore
from logic.benchmark.benchmark_suite import run_benchmarks
from logic.src.cli import parse_params
from logic.src.configs import Config
from logic.src.file_system import (
    delete_file_system_entries,
    perform_cryptographic_operations,
    update_file_system_entries,
)
from logic.test import PyTestRunner
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def run_test_suite(opts):
    """
    Execute the project's test suite based on provided options.

    Args:
        opts (dict): A dictionary containing test configuration parameters:
            - test_dir (str): Directory containing tests.
            - list (bool): If True, lists available test modules.
            - list_tests (bool): If True, lists collected tests.
            - module (list): Specific modules to run.
            - test_class (str): Specific class to run.
            - test_method (str): Specific method to run.
            - verbose (int): Verbosity level (0-2).
            - coverage (bool): Enable coverage reporting.
            - markers (str): Filter tests by markers.
            - failed_first (bool): Run previously failed tests first.
            - maxfail (int): Stop after N failures.
            - capture (str): Stack capture method ('fd', 'sys', 'no').
            - tb (str): Traceback style.
            - parallel (bool): Enable parallel execution.
            - keyword (str): Filter by keyword expression.

    Returns:
        int: Exit code (0 for success, non-zero for failure).

    Raises:
        Exception: If test execution encounters a fatal error not handled by the runner.
    """
    try:
        # Initialize test runner
        runner = PyTestRunner(test_dir=opts["test_dir"])

        # Handle information commands
        if opts["list"]:
            runner.list_modules()
            return 0

        if opts["list_tests"]:
            runner.list_tests(opts["module"][0] if opts["module"] else None)
            return 0

        # Run tests
        return runner.run_tests(
            modules=opts["module"],
            test_class=opts["test_class"],
            test_method=opts["test_method"],
            verbose=opts["verbose"],
            coverage=opts["coverage"],
            markers=opts["markers"],
            failed_first=opts["failed_first"],
            maxfail=opts["maxfail"],
            capture=opts["capture"],
            tb_style=opts["tb"],
            parallel=opts["parallel"],
            keyword=opts["keyword"],
        )
    except Exception as e:
        raise Exception(f"failed to run test suite due to {repr(e)}") from e


def pretty_print_args(comm, opts, inner_comm=None):
    """
    Format and print dictionary arguments in a clean, readable structure.

    This function utilizes `pprint` to format the options dictionary and then
    applies custom string manipulation to present it as a labeled block.

    Args:
        comm (str): The primary command name (e.g., 'train', 'gen_data').
        opts (dict): The dictionary of options/arguments to print.
        inner_comm (str, optional): A sub-command name (e.g., 'update' for 'file_system').
            Defaults to None.

    Raises:
        Exception: If formatting or printing fails.
    """
    try:
        # Capture the pprint output
        buffer = io.StringIO()
        printer = pprint.PrettyPrinter(width=1, indent=1, sort_dicts=False, stream=buffer)
        printer.pprint(opts)
        output = buffer.getvalue()

        # Pretty print the run options
        lines = output.splitlines()
        lines[0] = lines[0].lstrip("{")
        lines[-1] = lines[-1].rstrip("}")
        formatted = (
            comm
            + "{}".format(f" {inner_comm}" if inner_comm is not None else "")
            + ": {\n"
            + "\n".join(f" {line}" for line in lines)
            + "\n}"
        )
        print(formatted, end="\n\n")
    except Exception as e:
        raise Exception(f"failed to pretty print arguments due to {repr(e)}") from e


def pretty_print_hydra_config(cfg: DictConfig, filter_keys: list = None) -> None:  # type: ignore[assignment]
    """
    Pretty print filtered sections of the Hydra configuration.

    Args:
        cfg: The full Hydra configuration.
        filter_keys: List of top-level keys to include (e.g., ["train", "model", "env"]).
                     If None, prints the full config.
    """
    print("\n" + "=" * 80)
    print("HYDRA CONFIGURATION".center(80))
    print("=" * 80)

    # If filters are provided, create a subset of the config
    display_cfg = OmegaConf.masked_copy(cfg, filter_keys) if filter_keys else cfg

    # resolve=False to avoid interpolation errors with structured configs
    print(OmegaConf.to_yaml(display_cfg, resolve=False))
    print("=" * 80 + "\n")


def main(args) -> None:
    """
    Main dispatch function for the application.

    Routes execution to the appropriate sub-system based on the command provided
    in `args`. Handles top-level exception reporting and clean exit procedures.

    Args:
        args (tuple): A tuple returning from `parse_params`, containing:
            - comm (str or tuple): The command string or (command, sub_command) tuple.
            - opts (dict): The dictionary of configuration options.

    Sys.Exit:
        Exits with code 0 on success, 1 on error.
    """
    comm, opts = args
    inner_comm = None
    exit_code = 0
    try:
        if isinstance(comm, tuple) and len(comm) > 1:
            comm, inner_comm = comm
            pretty_print_args(comm, opts, inner_comm)
            assert comm == "file_system"
            if inner_comm == "update":
                update_file_system_entries(opts)
            elif inner_comm == "delete":
                delete_file_system_entries(opts)
            else:
                assert inner_comm == "cryptography"
                perform_cryptographic_operations(opts)
        else:
            pretty_print_args(comm, opts, inner_comm)
            if comm == "gui":
                exit_code = run_app_gui(opts)
            elif comm == "benchmark":
                run_benchmarks(opts)
            else:
                assert comm == "test_suite"
                run_test_suite(opts)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print("\n" + str(e))
        exit_code = 1
    finally:
        print(
            "\nFinished {}{} command execution with exit code: {}".format(
                comm, f" ({inner_comm}) " if inner_comm is not None else "", exit_code
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)


def main_dispatch() -> None:
    """
    Dispatch commands to either Hydra-based or legacy argparse-based systems.

    This function routes incoming CLI commands to the appropriate execution path
    based on the command type. It handles the transition between the old argparse
    system and the modern Hydra configuration system.

    System Architecture:
        - Hydra Commands: Configuration-driven, composable, supports sweeps
        - Legacy Commands: Direct argument parsing, simpler but less flexible

    Args:
        None (reads from sys.argv)

    Returns:
        None (calls sys.exit with appropriate code)
    """
    # ========================================================================
    # Dual Dispatch System
    # ========================================================================
    # This application uses TWO command routing systems for historical reasons:
    #
    # 1. HYDRA_COMMANDS (below): Modern commands routed through Hydra/Lightning.
    #    Examples: train, eval, test_sim, gen_data
    #    These bypass parse_params() and go directly to hydra_entry_point().
    #
    # 2. Legacy CLI (parse_params): Traditional argparse-based routing.
    #    Examples: gui, test_suite, file_system
    #    These use the original argument parser and main() function.
    #
    # The sys.argv manipulation below converts user commands into Hydra-compatible
    # overrides (e.g., 'eval' -> 'task=eval').
    #
    # MIGRATION NOTE: The long-term goal is to migrate ALL commands to Hydra for
    # consistency and to enable powerful features like config composition, sweeps,
    # and reproducibility. Priority migration candidates:
    #   - gui -> Hydra with GUI-specific config group
    #   - test_suite -> Hydra with test config group
    #   - file_system -> Hydra with filesystem operations config
    # ========================================================================
    HYDRA_COMMANDS = [
        "train",
        "mrl_train",
        "hp_optim",
        "eval",
        "test_sim",
        "gen_data",
    ]

    if len(sys.argv) > 1 and sys.argv[1] in HYDRA_COMMANDS:
        # Map command to task override
        command = sys.argv[1]
        if command in ["train", "mrl_train", "hp_optim"]:
            # These might have specific handling needs or default to train task
            pass

        if command in ["eval", "test_sim", "gen_data"]:
            sys.argv.append(f"task={command}")
            sys.argv.pop(1)  # Remove the command
        elif command == "mrl_train":
            sys.argv.append("rl.use_meta=True")
            sys.argv.pop(1)  # Remove the command
        else:  # For other commands like "train", "train_hydra", etc.
            sys.argv.pop(1)  # Remove the command

        hydra_entry_point()
    else:
        main(parse_params())


@hydra.main(version_base=None, config_path="assets/configs", config_name="config")
def hydra_entry_point(cfg: Config) -> float:
    """Unified entry point."""
    if cfg.task == "train":
        from logic.src.pipeline.features.train import run_hpo, run_training

        if cfg.verbose:
            training_keys = ["env", "model", "train", "rl", "optim"]
            pretty_print_hydra_config(cfg, filter_keys=training_keys)  # type: ignore[arg-type]

        if cfg.hpo.n_trials > 0:
            return run_hpo(cfg)
        else:
            return run_training(cfg)
    elif cfg.task == "eval":
        from logic.src.pipeline.features.base import flatten_config_dict
        from logic.src.pipeline.features.eval import run_evaluate_model, validate_eval_args

        # Convert Hydra config to dict
        eval_args = cast(dict[str, Any], OmegaConf.to_container(cfg.eval, resolve=True))
        # Flatten
        eval_args = flatten_config_dict(eval_args)
        # Validate and run
        args = validate_eval_args(eval_args)
        run_evaluate_model(args)
        return 0.0
    elif cfg.task == "test_sim":
        from logic.src.pipeline.features.base import flatten_config_dict
        from logic.src.pipeline.features.test import run_wsr_simulator_test, validate_test_sim_args

        # Convert Hydra config to dict
        sim_args = cast(dict[str, Any], OmegaConf.to_container(cfg.sim, resolve=True))
        # Flatten
        sim_args = flatten_config_dict(sim_args)
        # Validate and run
        args = validate_test_sim_args(sim_args)
        run_wsr_simulator_test(args)
        return 0.0
    elif cfg.task == "gen_data":
        from logic.src.data.generators import generate_datasets, validate_gen_data_args
        from logic.src.pipeline.features.base import flatten_config_dict

        # Convert Hydra config to dict
        data_args = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
        # Flatten
        data_args = flatten_config_dict(data_args)
        # Validate and run
        args = validate_gen_data_args(data_args)
        generate_datasets(args)
        return 0.0
    else:
        raise ValueError(f"Unknown task: {cfg.task}")


if __name__ == "__main__":
    main_dispatch()
