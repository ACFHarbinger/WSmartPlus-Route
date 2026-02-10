"""base.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import base
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import SimulationContext


class SimState(ABC):
    """Abstract base class for simulation states."""

    context: SimulationContext

    @abstractmethod
    def handle(self, ctx: SimulationContext) -> None:
        """Handles the current state and returns the next state."""
        pass
