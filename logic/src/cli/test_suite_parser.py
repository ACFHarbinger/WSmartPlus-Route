"""
Test suite related argument parsers.
"""

from logic.src.utils.definitions import TEST_MODULES


def add_test_suite_args(parser):
    """
    Adds all arguments related to the test suite to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added test suite arguments.
    """
    # Test selection
    parser.add_argument(
        "-m",
        "--module",
        nargs="+",
        choices=list(TEST_MODULES.keys()),
        help="Specific test module(s) to run",
    )
    parser.add_argument(
        "-c",
        "--class",
        dest="test_class",
        help="Specific test class to run (e.g., TestTrainCommand)",
    )
    parser.add_argument(
        "-t",
        "--test",
        dest="test_method",
        help="Specific test method to run (e.g., test_train_default_parameters)",
    )
    parser.add_argument("-k", "--keyword", help="Run tests matching the given keyword expression")
    parser.add_argument("--markers", help="Run tests matching the given marker expression")

    # Test execution options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument(
        "--ff",
        "--failed-first",
        dest="failed_first",
        action="store_true",
        help="Run failed tests first",
    )
    parser.add_argument(
        "-x",
        "--exitfirst",
        dest="maxfail",
        action="store_const",
        const=1,
        help="Exit on first failure",
    )
    parser.add_argument("--maxfail", type=int, help="Exit after N failures")
    parser.add_argument(
        "--tb",
        choices=["auto", "long", "short", "line", "native", "no"],
        default="auto",
        help="Traceback print mode",
    )
    parser.add_argument(
        "--capture",
        choices=["auto", "no", "sys", "fd"],
        default="auto",
        help="Capture mode for output",
    )
    parser.add_argument(
        "-n",
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    # Information commands
    parser.add_argument("-l", "--list", action="store_true", help="List all available test modules")
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all tests in specified module(s) or all tests",
    )
    parser.add_argument(
        "--test-dir",
        default="tests",
        help="Directory containing test files (default: tests)",
    )
    return parser


def validate_test_suite_args(args):
    """
    Validates and post-processes arguments for test_suite.
    """
    args = args.copy()
    return args
