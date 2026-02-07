"""
Centralized registry for the WSmart+ Route CLI parser.
"""

from logic.src.cli.base import ConfigsParser
from logic.src.cli.benchmark_parser import add_benchmark_args
from logic.src.cli.fs_parser import add_files_args
from logic.src.cli.gui_parser import add_gui_args
from logic.src.cli.ts_parser import add_test_suite_args


def get_parser() -> ConfigsParser:
    """
    Creates and returns the main ConfigsParser with all subcommands registered.
    """
    parser = ConfigsParser(description="WSmart+ Route Unified CLI Framework")
    subparsers = parser.add_subparsers(dest="command", help="The command to execute", required=True)

    # Files
    files_parser = subparsers.add_parser("file_system", help="File system operations")
    add_files_args(files_parser)

    # GUI
    gui_p = subparsers.add_parser("gui", help="Launch the GUI")
    add_gui_args(gui_p)

    # Test Suite
    ts_parser = subparsers.add_parser("test_suite", help="Run the test suite")
    add_test_suite_args(ts_parser)

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    add_benchmark_args(bench_parser)

    # TUI

    return parser
