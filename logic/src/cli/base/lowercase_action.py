import argparse
from typing import Any, Optional


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
