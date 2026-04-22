"""
Controller module.

This module provides the entry points for the WSmart-Route application.
It handles the dispatching of commands to the appropriate handlers.

Attributes:
    parser_entry_point: Entry point for the parser-based CLI.
    hydra_entry_point: Entry point for the Hydra-based CLI.

Example:
    >>> from logic.controller import parser_entry_point, hydra_entry_point
    >>> parser_entry_point()
    # Runs the default command (gui) with default configuration
    >>> hydra_entry_point("--task=train --train.epochs=10")
    # Runs the train task with 10 epochs
"""

from .hydra_dispatch import hydra_entry_point
from .parser_dispatch import parser_entry_point

__all__ = [
    "parser_entry_point",
    "hydra_entry_point",
]
