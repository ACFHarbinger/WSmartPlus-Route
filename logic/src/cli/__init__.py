"""
Unified entry point for the WSmart+ Route modular CLI.
"""

import argparse
from logic.src.cli.base_parser import ConfigsParser, LowercaseAction, StoreDictKeyPair, UpdateFunctionMapActionFactory
from logic.src.cli.train_parser import add_train_args, add_mrl_train_args, add_hp_optim_args, validate_train_args
from logic.src.cli.sim_parser import add_test_sim_args, add_eval_args, validate_test_sim_args, validate_eval_args
from logic.src.cli.data_parser import add_gen_data_args, validate_gen_data_args
from logic.src.cli.fs_parser import add_files_args, validate_file_system_args
from logic.src.cli.gui_parser import add_gui_args, validate_gui_args
from logic.src.cli.test_suite_parser import add_test_suite_args, validate_test_suite_args

def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    subparsers = parser.add_subparsers(dest="command", help="The command to execute", required=True)

    # Training
    train_parser = subparsers.add_parser("train", help="Generic training for neural model")
    add_train_args(train_parser)

    # MRL Train
    mrl_parser = subparsers.add_parser("mrl_train", help="Meta-RL training")
    add_mrl_train_args(mrl_parser)

    # HPO
    hpo_parser = subparsers.add_parser("hp_optim", help="Hyperparameter optimization")
    add_hp_optim_args(hpo_parser)

    # Gen Data
    gen_parser = subparsers.add_parser("gen_data", help="Generate training data")
    add_gen_data_args(gen_parser)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    add_eval_args(eval_parser)

    # Test Sim
    sim_parser = subparsers.add_parser("test_sim", help="Run simulation test")
    add_test_sim_args(sim_parser)

    # Files
    files_parser = subparsers.add_parser("file_system", help="File system operations")
    add_files_args(files_parser)

    # GUI
    gui_p = subparsers.add_parser("gui", help="Launch the GUI")
    add_gui_args(gui_p)

    # Test Suite
    ts_parser = subparsers.add_parser("test_suite", help="Run the test suite")
    add_test_suite_args(ts_parser)

    return parser

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
        if command in ["train", "mrl_train", "hp_optim"]:
            opts = validate_train_args(opts)
        elif command == "gen_data":
            opts = validate_gen_data_args(opts)
        elif command == "eval":
            opts = validate_eval_args(opts)
        elif command == "test_sim":
            opts = validate_test_sim_args(opts)
        elif command == "file_system":
            # This returns a tuple: (fs_command, validated_opts)
            command, opts = validate_file_system_args(opts)
            command = ("file_system", command)  # Re-wrap for main() function handling
        elif command == "gui":
            opts = validate_gui_args(opts)
        elif command == "test_suite":
            opts = validate_test_suite_args(opts)
        return command, opts
    except (argparse.ArgumentError, AssertionError) as e:
        parser.error_message(f"Error: {e}", print_help=True)
    except Exception as e:
        parser.error_message(f"An unexpected error occurred: {e}", print_help=False)
