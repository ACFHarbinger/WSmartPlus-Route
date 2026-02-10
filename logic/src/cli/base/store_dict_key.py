"""store_dict_key.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import store_dict_key
"""

import argparse
from typing import Any, Optional, Sequence, Union


class StoreDictKeyPair(argparse.Action):
    """Custom action to parse key=value into a dictionary."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Union[str, Sequence[Any], None],
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
        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]

        for kv in values:
            if "=" in kv:
                k, v = kv.split("=", 1)
                my_dict[k] = v
            else:
                raise argparse.ArgumentError(self, f"Could not parse argument '{kv}' as key=value format")
        setattr(namespace, self.dest, my_dict)
