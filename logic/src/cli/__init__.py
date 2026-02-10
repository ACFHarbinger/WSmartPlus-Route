"""
Unified entry point for the WSmart+ Route modular CLI.
"""

import argparse

from logic.src.cli.benchmark_parser import validate_benchmark_args
from logic.src.cli.fs_parser import add_files_args, validate_file_system_args
from logic.src.cli.gui_parser import add_gui_args, validate_gui_args
from logic.src.cli.registry import get_parser
from logic.src.cli.ts_parser import add_test_suite_args, validate_test_suite_args

from .base import (
    ConfigsParser,
    LowercaseAction,
    StoreDictKeyPair,
    UpdateFunctionMapActionFactory,
)


def parse_params():
    """
    Parses arguments, determines the command, and performs necessary validation.
    Returns: (command, validated_opts) where 'command' might be a tuple (comm, inner_comm)
    """
    parser = get_parser()

    try:
        # Parse arguments into a dictionary using the custom handler
        command, opts = parser.parse_process_args()

        # --- COMMAND-SPECIFIC VALIDATION AND POST-PROCESSING ---
        # "gen_data", "eval", "test_sim", "train" are now fully handled by Hydra or removed from legacy flow
        if command == "file_system":
            # This returns a tuple: (fs_command, validated_opts)
            command, opts = validate_file_system_args(opts)
            command = ("file_system", command)  # type: ignore[assignment]  # Re-wrap for main()
        elif command == "gui":
            opts = validate_gui_args(opts)
        elif command == "test_suite":
            opts = validate_test_suite_args(opts)
        elif command == "benchmark":
            opts = validate_benchmark_args(opts)
        return command, opts
    except (argparse.ArgumentError, AssertionError) as e:
        parser.error_message(f"Error: {e}", print_help=True)
    except Exception as e:
        parser.error_message(f"An unexpected error occurred: {e}", print_help=False)
