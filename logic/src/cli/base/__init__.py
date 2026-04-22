"""
Base CLI module.

Attributes:
    ConfigsParser: Custom ArgumentParser to handle string-based nargs correctly.
    UpdateFunctionMapActionFactory: Factory for mapping string update functions.
    LowercaseAction: Action to convert argument value to lowercase.
    StoreDictKeyPair: Custom action to parse key=value into a dictionary.

Example:
    >>> from logic.src.cli.base import ConfigsParser, UpdateFunctionMapActionFactory, LowercaseAction, StoreDictKeyPair
    >>> parser = ConfigsParser()
    >>> parser.add_argument('--test', action='store_true')
    >>> parser.parse_args(['--test'])
    Namespace(test=True)
"""

from .lowercase_action import LowercaseAction
from .parser import ConfigsParser
from .store_dict_key import StoreDictKeyPair
from .update_function_factory import UpdateFunctionMapActionFactory

__all__ = [
    "ConfigsParser",
    "UpdateFunctionMapActionFactory",
    "LowercaseAction",
    "StoreDictKeyPair",
]
