"""
Lowercase action module.

Attributes:
    LowercaseAction: Action to convert argument value to lowercase.

Example:
    >>> from logic.src.cli.base.lowercase_action import LowercaseAction
    >>> action = LowercaseAction(option_strings=['--name'], dest='name')
    >>> namespace = argparse.Namespace()
    >>> action(None, namespace, 'Test')
    >>> namespace.name
    'test'
"""

import argparse
from typing import Any, Optional


class LowercaseAction(argparse.Action):
    """
    Action to convert argument value to lowercase.

    Attributes:
        dest: Destination attribute in the namespace.
        nargs: Number of arguments.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """
        Invoke action to lowercase input values.

        Args:
            parser: The argument parser.
            namespace: The namespace to store the result.
            values: The value provided in the command line.
            option_string: The option string, if any.
        """
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)
