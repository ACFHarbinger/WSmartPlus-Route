"""
Abstract Strategy Pattern for Weight Adjustment.

This module defines the base interface for all weight adjustment strategies used in
meta-learning. The WeightAdjustmentStrategy provides a common contract that ensures
all implementations can be used interchangeably in the training pipeline.

The strategy pattern allows the training orchestrator to be decoupled from the specific
weight adjustment algorithm, enabling easy experimentation with different meta-learning
approaches without modifying the core training logic.

Classes:
    WeightAdjustmentStrategy: Abstract base class defining the weight adjustment interface

Design Pattern:
    Strategy Pattern - defines a family of algorithms (weight adjustment strategies),
    encapsulates each one, and makes them interchangeable.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class WeightAdjustmentStrategy(ABC):
    """
    Interface for strategies that propose and update cost weights for the reinforcement learning environment.
    """

    @abstractmethod
    def propose_weights(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Returns the weights for the next simulation day or step.

        Args:
            context: Optional dictionary containing context information (e.g., current day, environment state).

        Returns:
            A dictionary mapping weight names to their proposed values.
        """
        pass

    @abstractmethod
    def feedback(
        self,
        reward: float,
        metrics: Dict[str, float],
        day: int = None,
        step: int = None,
    ):
        """
        Receives feedback from the environment to update internal state.

        Args:
            reward: The scalar reward received.
            metrics: A dictionary of performance metrics (components of the cost/reward).
            day: The current simulation day (optional).
            step: The current training step (optional).
        """
        pass

    @abstractmethod
    def get_current_weights(self) -> Dict[str, float]:
        """
        Returns the currently held weights.
        """
        pass
