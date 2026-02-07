"""
Parser arguments for the Benchmark Suite.
"""

from typing import Any, Dict

from logic.src.cli.base import ConfigsParser


def add_benchmark_args(parser: ConfigsParser) -> None:
    """
    Adds arguments for the benchmark command.
    """
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        choices=["all", "neural", "ls", "solvers"],
        help="Subset of benchmarks to run (neural, ls, solvers, or all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON.",
    )


def validate_benchmark_args(opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates benchmark arguments.
    """
    # currently no complex validation needed, just return opts
    return opts
