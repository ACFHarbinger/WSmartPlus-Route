"""
Update function factory module.

Attributes:
    UpdateFunctionMapActionFactory: Factory for mapping string update functions.

Example:
    >>> from logic.src.cli.base.update_function_factory import UpdateFunctionMapActionFactory
    >>> UpdateFunctionMapActionFactory()
    <class 'logic.src.cli.base.update_function_factory.UpdateFunctionMapActionFactory.<locals>.UpdateFunctionMapAction'>
"""

import argparse
from typing import Any, List, Optional, Union

from logic.src.constants.stats import STATS_FUNCTION_MAP
from logic.src.constants.system import OPERATION_MAP


def UpdateFunctionMapActionFactory(inplace: bool = False) -> type:
    """Factory for mapping string update functions.

    Args:
        inplace: Whether to use the in-place operation map or the stats function map.

    Returns:
        A class type for UpdateFunctionMapAction.
    """

    class UpdateFunctionMapAction(argparse.Action):
        """
        Action that maps input strings to update functions.

        Attributes:
            inplace: Whether to use the in-place operation map or the stats function map.
        """

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
                kwargs: Additional keyword arguments.
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

            Returns:
                None
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
