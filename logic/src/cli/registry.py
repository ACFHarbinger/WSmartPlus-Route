"""
Centralized registry for the WSmart+ Route CLI parser.
"""

from logic.src.cli.base_parser import ConfigsParser
from logic.src.cli.data_parser import add_gen_data_args
from logic.src.cli.fs_parser import add_files_args
from logic.src.cli.gui_parser import add_gui_args
from logic.src.cli.sim_parser import add_eval_args, add_test_sim_args
from logic.src.cli.ts_parser import add_test_suite_args


def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    subparsers = parser.add_subparsers(dest="command", help="The command to execute", required=True)

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
