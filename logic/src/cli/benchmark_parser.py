"""
Parser arguments for the Benchmark Suite.

Attributes:
    add_benchmark_args: Adds arguments for the benchmark command
    validate_benchmark_args: Validates benchmark arguments

Example:
    >>> from logic.src.cli.benchmark_parser import add_benchmark_args, validate_benchmark_args
    >>> parser = ConfigsParser()
    >>> add_benchmark_args(parser)
    >>> validate_benchmark_args(parser.parse_args())
    {'subset': 'all', 'device': 'auto', 'output': None}
"""

from typing import Any, Dict

from logic.src.cli.base import ConfigsParser


def add_benchmark_args(parser: ConfigsParser) -> None:
    """
    Adds arguments for the benchmark command.

    Args:
        parser: ConfigsParser instance

    Returns:
        None
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

    Args:
        opts: Dictionary of benchmark arguments

    Returns:
        Dictionary of validated benchmark arguments
    """
    # currently no complex validation needed, just return opts
    return opts
