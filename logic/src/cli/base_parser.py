"""
Base parser utilities for the WSmart+ Route modular CLI.

This module provides the core ConfigsParser class and shared argument actions
used across all command-specific parsers.
"""

import argparse
import sys
from typing import Sequence
from multiprocessing import cpu_count
from logic.src.utils.functions import parse_softmax_temperature
from logic.src.utils.definitions import (
    OPERATION_MAP,
    STATS_FUNCTION_MAP,
)

class ConfigsParser(argparse.ArgumentParser):
    """
    Custom ArgumentParser to handle string-based nargs correctly.
    """

    def _str_to_nargs(self, nargs):
        """
        Converts a single string argument into a list if expected by nargs.
        """
        if isinstance(nargs, Sequence) and len(nargs) == 1:
            return nargs[0].split() if isinstance(nargs[0], str) else nargs
        else:
            return nargs

    def _process_args(self, namespace):
        """
        Post-processes arguments in the namespace, handling special narg conversions.
        """
        for action in self._actions:
            if action.nargs is not None:
                if action.dest == "help":
                    continue
                value = getattr(namespace, action.dest)
                if value is not None:
                    transformed_value = self._str_to_nargs(value)
                    setattr(namespace, action.dest, transformed_value)

    def parse_command(self, args=None):
        """
        Parses only the command from the arguments.
        """
        if args is None:
            args = sys.argv[1:]
        namespace = super().parse_args(args)
        return getattr(namespace, "command", None)

    def parse_process_args(self, args=None, command=None):
        """
        Parses arguments and returns the command and options dictionary.
        """
        if args is None:
            args = sys.argv[1:]

        actions_to_check = list(self._actions)

        command_name = None
        if args and not args[0].startswith("-"):
            command_name = args[0]

        if command_name:
            subparsers_action = next(
                (
                    a
                    for a in actions_to_check
                    if isinstance(a, argparse._SubParsersAction)
                ),
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
        filtered_args = {
            key: value if value != "" else None
            for key, value in parsed_args_dict.items()
        }

        command = filtered_args.pop("command")
        return command, filtered_args

    def error_message(self, message, print_help=True):
        """
        Prints error message and optionally help.
        """
        print(message, end=" ")
        if print_help:
            self.print_help()
        raise

class LowercaseAction(argparse.Action):
    """Action to convert argument value to lowercase."""
    def __call__(self, parser, namespace, values, option_string=None):
        """Invoke action: lowercase input string."""
        if values is not None:
            values = str(values).lower()
        setattr(namespace, self.dest, values)

class StoreDictKeyPair(argparse.Action):
    """Custom action to parse key=value into a dictionary."""
    def __call__(self, parser, namespace, values, option_string=None):
        """Invoke action: parse key=value strings into dictionary."""
        my_dict = {}
        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", 1)
                my_dict[k] = v
            else:
                raise argparse.ArgumentError(
                    self, f"Could not parse argument '{kv}' as key=value format"
                )
        setattr(namespace, self.dest, my_dict)

def UpdateFunctionMapActionFactory(inplace=False):
    """Factory for mapping string update functions."""
    class UpdateFunctionMapAction(argparse.Action):
        """Action that maps input strings to update functions."""
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            """Initialize the update function map action."""
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)
            self.inplace = inplace
        def __call__(self, parser, namespace, values, option_string=None):
            """Invoke action: map string to update function."""
            if values is not None:
                if self.inplace:
                    values = OPERATION_MAP.get(str(values).replace(" ", ""), None)
                else:
                    values = STATS_FUNCTION_MAP.get(str(values).replace(" ", ""), None)
            if values is None:
                raise ValueError(f"Invalid update function: {values}")
            setattr(namespace, self.dest, values)
    return UpdateFunctionMapAction
