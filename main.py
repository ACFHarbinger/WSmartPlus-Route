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

import sys
import warnings

from logic.hydra_dispatch import hydra_entry_point
from logic.parser_dispatch import parser_entry_point
from logic.src.cli import parse_params

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="jax.tree_util.register_keypaths is deprecated",
)


def main() -> None:
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
    # ========================================================================
    HYDRA_TASKS = {"train", "evaluation", "eval", "test_sim", "gen_data", "hpo", "meta_train"}
    if len(sys.argv) > 1 and sys.argv[1] in HYDRA_TASKS:
        task = sys.argv.pop(1)
        sys.argv.append(f"tasks={task}")
        sys.argv.append(f"task={task}")
        hydra_entry_point()
    else:
        parser_entry_point(parse_params())


if __name__ == "__main__":
    main()
