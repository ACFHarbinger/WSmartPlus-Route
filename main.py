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
import multiprocessing as mp
import os
import pprint
import signal
import sys
import traceback
import warnings

import logic.src.constants as udef
from gui.src.app import launch_results_window, run_app_gui
from logic.src.cli import parse_params
from logic.src.data.generate_data import generate_datasets
from logic.src.file_system import (
    delete_file_system_entries,
    perform_cryptographic_operations,
    update_file_system_entries,
)
from logic.src.pipeline.features.eval import run_evaluate_model
from logic.src.pipeline.features.test import run_wsr_simulator_test
from logic.test import PyTestRunner

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)


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
        raise Exception(f"failed to run test suite due to {repr(e)}")


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
        printer = pprint.PrettyPrinter(width=1, indent=1, sort_dicts=False)
        buffer = io.StringIO()
        printer._stream = buffer  # Redirect PrettyPrinter's internal stream
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
        raise Exception(f"failed to pretty print arguments due to {repr(e)}")


def main(args):
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
            inner_comm = None
            pretty_print_args(comm, opts, inner_comm)
            if comm == "gui":
                exit_code = run_app_gui(opts)
            elif comm == "test_suite":
                run_test_suite(opts)
            else:
                if comm == "gen_data":
                    generate_datasets(opts)
                elif comm == "eval":
                    run_evaluate_model(opts)
                elif comm == "test_sim":
                    if opts["real_time_log"]:
                        mp.set_start_method("spawn", force=True)
                        simulation_process = mp.Process(target=run_wsr_simulator_test, args=(opts,))
                        log_path = os.path.join(
                            udef.ROOT_DIR,
                            "assets",
                            opts["output_dir"],
                            str(opts["days"]) + "_days",
                            str(opts["area"]) + "_" + str(opts["size"]),
                            f"log_realtime_{opts['data_distribution']}_{opts['n_samples']}N.jsonl",
                        )
                        simulation_process.start()

                        # Define the handler function that terminates the subprocess
                        def handle_interrupt(signum, frame):
                            """
                            Handle SIGINT (Ctrl+C) during real-time simulation.

                            Terminates the simulation subprocess and forces a clean exit
                            to prevent zombie processes or hung UIs.

                            Args:
                                signum (int): The signal number.
                                frame (frame): The current stack frame.
                            """
                            print("\nCtrl+C received. Terminating simulation process...")
                            if simulation_process.is_alive():
                                simulation_process.terminate()
                                simulation_process.join()
                            # Force the GUI application to quit gracefully
                            sys.exit(0)

                        # Register the handler only for this scope
                        original_sigint_handler = signal.getsignal(signal.SIGINT)
                        signal.signal(signal.SIGINT, handle_interrupt)

                        try:
                            # 3. Blocking GUI call
                            exit_code = launch_results_window(opts["policies"], log_path)

                        except SystemExit as e:
                            # Catch the sys.exit(0) from the handler, if triggered.
                            exit_code = e.code

                        finally:
                            # 4. Restore original handler and clean up the process
                            signal.signal(signal.SIGINT, original_sigint_handler)

                            if simulation_process.is_alive():
                                print("GUI closed. Terminating lingering simulation process.")
                                simulation_process.terminate()
                                simulation_process.join()
                    else:
                        run_wsr_simulator_test(opts)
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


if __name__ == "__main__":
    # ========================================================================
    # Dual Dispatch System
    # ========================================================================
    # This application uses TWO command routing systems for historical reasons:
    #
    # 1. HYDRA_COMMANDS (below): Modern commands routed through Hydra/Lightning.
    #    Examples: train, eval, test_sim, gen_data
    #    These bypass parse_params() and go directly to unified_main().
    #
    # 2. Legacy CLI (parse_params): Traditional argparse-based routing.
    #    Examples: gui, test_suite, file_system
    #    These use the original argument parser and main() function.
    #
    # The sys.argv manipulation below (lines 296-304) converts user commands
    # into Hydra-compatible overrides (e.g., 'eval' -> 'task=eval').
    # ========================================================================
    HYDRA_COMMANDS = [
        "train",
        "train_hydra",
        "train_lightning",
        "mrl_train",
        "hp_optim",
        "hp_optim_hydra",
        "eval",
        "test_sim",
        "gen_data",
    ]

    if len(sys.argv) > 1 and sys.argv[1] in HYDRA_COMMANDS:
        # Map command to task override
        command = sys.argv[1]

        # Inject task override if needed, handling the case where it might already be specified
        task_override = f"task={command}" if command in ["eval", "test_sim", "gen_data", "train"] else None

        # For legacy train commands like 'train_hydra', default to 'task=train' (already default in config)
        if command in ["train_hydra", "train_lightning", "mrl_train", "hp_optim", "hp_optim_hydra"]:
            # These might have specific handling needs or default to train task
            pass

        # Bypass legacy parsing and delegate to Hydra/Lightning pipeline
        from logic.src.pipeline.features.train import main as unified_main

        if command in ["eval", "test_sim", "gen_data"]:
            sys.argv.append(f"task={command}")
            sys.argv.pop(1)  # Remove the command
        elif command == "mrl_train":
            sys.argv.append("rl.use_meta=True")
            sys.argv.pop(1)  # Remove the command
        else:  # For other commands like "train", "train_hydra", etc.
            sys.argv.pop(1)  # Remove the command

        unified_main()
    else:
        main(parse_params())
