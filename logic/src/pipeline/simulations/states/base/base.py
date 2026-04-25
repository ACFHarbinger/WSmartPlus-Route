"""
Abstract Base Class for Simulation States.

This module defines the SimState interface for the Simulation State Machine.

Attributes:
    SimState: Abstract base class for all simulation states.

Example:
    >>> # from logic.src.pipeline.simulations.states.base.base import SimState
    >>> # class MyState(SimState): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import SimulationContext


class SimState(ABC):
    """
    Abstract base class for simulation states.

    Attributes:
        context: The SimulationContext object managing the state machine.
    """

    context: SimulationContext

    @abstractmethod
    def handle(self, ctx: SimulationContext) -> None:
        """
        Handles the logic for the current state and triggers transitions.

        Args:
            ctx: The simulation context object.
        """
        pass
