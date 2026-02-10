"""
GUI related argument parsers.
"""

from logic.src.cli.base import LowercaseAction
from logic.src.constants import APP_STYLES


def add_gui_args(parser):
    """
    Adds all arguments related to the GUI to the given parser.

    Args:
        parser: The argparse parser or subparser.

    Returns:
        The parser with added GUI arguments.
    """
    parser.add_argument(
        "--app_style",
        action=LowercaseAction,
        type=str,
        default="fusion",
        help="Style for the GUI application",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Test mode for the GUI (commands are only printed, not executed).",
    )
    return parser


def validate_gui_args(args):
    """
    Validates and post-processes arguments for gui.src.
    """
    args = args.copy()
    assert args.get("app_style") in [None] + APP_STYLES, (
        f"Invalid application style '{args.get('app_style')}' - app_style value must be: {[None] + APP_STYLES}"
    )
    return args
