"""
Base parser utilities for the WSmart+ Route modular CLI.

This module provides the core ConfigsParser class and shared argument actions
used across all command-specific parsers.
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from logic.src.constants import (
    OPERATION_MAP,
    STATS_FUNCTION_MAP,
)


class ConfigsParser(argparse.ArgumentParser):
    """Custom ArgumentParser to handle string-based nargs correctly.

    This class extends ArgumentParser to provide additional functionality for
    processing command-line arguments, especially those involving sequence types
    and nested parser structures.
    """

    def _str_to_nargs(self, nargs: Union[str, Sequence]) -> Union[str, Sequence]:
        """Convert a single string argument into a list if expected by nargs.

        Args:
            nargs: The value of the argument, potentially a single-element sequence containing a string.

        Returns:
            The processed value, split into a list if it was a space-separated string within a sequence.
        """
        if isinstance(nargs, Sequence) and len(nargs) == 1:
            return nargs[0].split() if isinstance(nargs[0], str) else nargs
        else:
            return nargs

    def _process_args(self, namespace: argparse.Namespace) -> None:
        """Post-process arguments in the namespace, handling special narg conversions.

        Args:
            namespace: The namespace containing parsed arguments.
        """
        for action in self._actions:
            if action.nargs is not None:
                if action.dest == "help":
                    continue
                value = getattr(namespace, action.dest)
                if value is not None:
                    transformed_value = self._str_to_nargs(value)
                    setattr(namespace, action.dest, transformed_value)

    def parse_command(self, args: Optional[Sequence[str]] = None) -> Optional[str]:
        """Parse only the command from the arguments.

        Args:
            args: The sequence of arguments to parse. Defaults to sys.argv[1:].

        Returns:
            The name of the command found in the arguments, if any.
        """
        if args is None:
            args = sys.argv[1:]
        namespace = super().parse_args(args)
        return getattr(namespace, "command", None)

    def parse_process_args(
        self, args: Optional[List[str]] = None, command: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Parse arguments and returns the command and options dictionary.

        This method performs a more complex parsing that handles space-separated
        strings intended for multi-value arguments (nargs) by splitting them
        before full parsing.

        Args:
            args: The sequence of arguments to parse. Defaults to sys.argv[1:].
            command: Not used, kept for signature compatibility. Defaults to None.

        Returns:
            A tuple containing (command_name, dictionary_of_filtered_arguments).
        """
        if args is None:
            args = sys.argv[1:]

        actions_to_check = list(self._actions)

        command_name = None
        if args and not args[0].startswith("-"):
            command_name = args[0]

        if command_name:
            subparsers_action = next(
                (a for a in actions_to_check if isinstance(a, argparse._SubParsersAction)),
                None,
            )

            if subparsers_action and command_name in subparsers_action.choices:
                sub_parser = subparsers_action.choices[command_name]
                actions_to_check.extend(sub_parser._actions)

        for action in actions_to_check:
            if action.dest == "help":
                continue

            if action.nargs is not None and action.type is not None:
                opts = action.option_strings
                idx = next((i for i, x in enumerate(args) if x in opts), None)
                if idx is not None and (idx + 1) < len(args):
                    arg_val = args[idx + 1]
                    if isinstance(arg_val, str) and not arg_val.startswith("-"):
                        arg_parts = arg_val.split()
                        if len(arg_parts) > 1:
                            args[idx + 1 : idx + 2] = arg_parts

        subnamespace = super().parse_args(args)
        parsed_args_dict = vars(subnamespace)
        filtered_args = {key: value if value != "" else None for key, value in parsed_args_dict.items()}

        command = filtered_args.pop("command")
        return command, filtered_args

    def error_message(self, message: str, print_help: bool = True) -> None:
        """Print error message and optionally help.

        Args:
            message: The error message to display.
            print_help: Whether to print the help text as well. Defaults to True.

        Raises:
            The original exception if this is caught in a block, or simply exits the program via ArgumentParser mechanism.
        """
        print(message, end=" ")
        if print_help:
            self.print_help()
        raise


class LowercaseAction(argparse.Action):
    """Action to convert argument value to lowercase."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        """Invoke action to lowercase input values.

        Args:
            parser: The argument parser.
            namespace: The namespace to store the result.
            values: The value provided in the command line.
            option_string: The option string, if any.
        """
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)


class StoreDictKeyPair(argparse.Action):
    """Custom action to parse key=value into a dictionary."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: List[str],
        option_string: Optional[str] = None,
    ) -> None:
        """Invoke action to parse key=value strings into a dictionary.

        Args:
            parser: The argument parser.
            namespace: The namespace to store the result.
            values: A list of 'key=value' strings.
            option_string: The option string, if any.

        Raises:
            argparse.ArgumentError: If a string does not follow the 'key=value' format.
        """
        my_dict = {}
        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", 1)
                my_dict[k] = v
            else:
                raise argparse.ArgumentError(self, f"Could not parse argument '{kv}' as key=value format")
        setattr(namespace, self.dest, my_dict)


def UpdateFunctionMapActionFactory(inplace: bool = False) -> type:
    """Factory for mapping string update functions.

    Args:
        inplace: Whether to use the in-place operation map or the stats function map.

    Returns:
        A class type for UpdateFunctionMapAction.
    """

    class UpdateFunctionMapAction(argparse.Action):
        """Action that maps input strings to update functions."""

        def __init__(
            self,
            option_strings: List[str],
            dest: str,
            nargs: Optional[Union[int, str]] = None,
            **kwargs,
        ):
            """Initialize the update function map action.

            Args:
                option_strings: Option strings for this action.
                dest: Destination attribute in the namespace.
                nargs: Number of arguments.
                **kwargs: Additional keyword arguments.
            """
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)
            self.inplace = inplace

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: Optional[str] = None,
        ) -> None:
            """Invoke action: map string to update function.

            Args:
                parser: The argument parser.
                namespace: The namespace to store the result.
                values: The input string value(s).
                option_string: The option string, if any.

            Raises:
                ValueError: If the input value does not map to a valid function.
            """
            if values is not None:
                if self.inplace:
                    values = OPERATION_MAP.get(str(values).replace(" ", ""), None)
                else:
                    values = STATS_FUNCTION_MAP.get(str(values).replace(" ", ""), None)
            if values is None:
                raise ValueError(f"Invalid update function: {values}")
            setattr(namespace, self.dest, values)

    return UpdateFunctionMapAction
