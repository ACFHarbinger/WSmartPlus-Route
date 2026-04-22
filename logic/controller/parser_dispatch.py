"""
Parser dispatch module.

This module provides the unified entry point for all configuration-driven commands
parsed by the standard argparse-based CLI. It handles dispatching to the
appropriate handler based on the command-line arguments.

Attributes:
    parser_entry_point: Unified entry point for all configuration-driven commands.
    run_test_suite: Execute the project's test suite.
    pretty_print_args: Format and print dictionary arguments.

Example:
    >>> from logic.controller.parser_dispatch import parser_entry_point
    >>> parser_entry_point()
    # Runs the default command (gui) with default configuration
    >>> parser_entry_point("--command=benchmark --benchmark.num_instances=10")
    # Runs the benchmark command with specified number of instances
"""

import io
import pprint
import sys
import traceback

from logic.benchmark.benchmark_suite import run_benchmarks
from logic.src.file_system import (
    delete_file_system_entries,
    perform_cryptographic_operations,
    update_file_system_entries,
)
from logic.test import PyTestRunner


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


def parser_entry_point(args) -> None:
    """
    Unified parser entry point for all configuration-driven commands.

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
    if opts.get("profile"):
        from logic.src.tracking.profiling.profiler import start_global_profiling

        start_global_profiling(log_dir=opts.get("log_dir", "logs"))

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
                from gui.src.app import run_app_gui

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
