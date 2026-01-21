"""
Centralized registry for the WSmart+ Route CLI parser.
"""

from logic.src.cli.base_parser import ConfigsParser
from logic.src.cli.data_parser import add_gen_data_args
from logic.src.cli.fs_parser import add_files_args
from logic.src.cli.gui_parser import add_gui_args
from logic.src.cli.hpo_parser import add_hp_optim_args
from logic.src.cli.meta_train_parser import add_mrl_train_args
from logic.src.cli.sim_parser import add_eval_args, add_test_sim_args
from logic.src.cli.train_parser import add_train_args
from logic.src.cli.ts_parser import add_test_suite_args


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

    # TUI
    subparsers.add_parser("tui", help="Launch the Terminal User Interface")

    return parser
