"""
Base environment class for combinatorial optimization problems.

This module provides the foundation for RL4CO-style environment abstraction,
enabling unified state management via TensorDict and problem-agnostic interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from tensordict import TensorDict


class RL4COEnvBase(ABC):
    """
    Base environment class for combinatorial optimization problems.

    This class provides a unified interface for problem environments following
    the RL4CO architecture pattern. All problem-specific environments should
    inherit from this class.

    Attributes:
        name: Unique identifier for the environment type.
        generator: Optional data generator for creating problem instances.
        device: Device to place tensors on (cpu/cuda).
    """

    name: str = "base"

    def __init__(
        self,
        generator: Optional[Any] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base environment.

        Args:
            generator: Data generator instance for creating problem instances.
            generator_params: Parameters to pass to the generator if not provided.
            device: Device to place tensors on.
            **kwargs: Additional keyword arguments.
        """
        self.device = torch.device(device)
        self.generator = generator
        self.generator_params = generator_params or {}
        self._kwargs = kwargs

    @abstractmethod
    def _reset(self, td: TensorDict, batch_size: Optional[int] = None) -> TensorDict:
        """
        Initialize episode state from problem instance.

        This method should be implemented by subclasses to handle
        problem-specific state initialization.

        Args:
            td: TensorDict containing the problem instance data.
            batch_size: Optional batch size for generating new instances.

        Returns:
            TensorDict with initialized state fields.
        """
        raise NotImplementedError

    @abstractmethod
    def _step(self, td: TensorDict) -> TensorDict:
        """
        Execute action and return new state.

        This method should be implemented by subclasses to handle
        problem-specific state transitions.

        Args:
            td: TensorDict containing current state and action.

        Returns:
            TensorDict with updated state after action execution.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute final reward/cost for complete solution.

        This method should be implemented by subclasses to compute
        problem-specific rewards (typically called at episode end).

        Args:
            td: TensorDict containing the complete solution state.
            actions: Optional tensor of actions taken (tour sequence).

        Returns:
            Tensor of rewards for each instance in the batch.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Return mask of valid actions.

        This method should be implemented by subclasses to compute
        problem-specific action masking.

        Args:
            td: TensorDict containing current state.

        Returns:
            Boolean tensor indicating valid (True) and invalid (False) actions.
        """
        raise NotImplementedError

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """
        Check if episodes are complete.

        Default implementation checks if all nodes are visited or
        we've returned to depot. Override for problem-specific logic.

        Args:
            td: TensorDict containing current state.

        Returns:
            Boolean tensor indicating which episodes are done.
        """
        # Default: done when current node is depot and we've moved
        current_node = td.get("current_node", None)
        step_count = td.get("i", None)

        if current_node is None or step_count is None:
            return torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

        # Done if we're back at depot (node 0) after at least one step
        return (current_node.squeeze(-1) == 0) & (step_count.squeeze(-1) > 0)

    def reset(self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None) -> TensorDict:
        """
        Public reset with common preprocessing.

        This method handles the common reset logic and delegates
        problem-specific initialization to _reset().

        Args:
            td: Optional TensorDict containing problem instance.
                If None, generates new instances using the generator.
            batch_size: Batch size for generating new instances.

        Returns:
            TensorDict with initialized state ready for rollout.
        """
        # Generate new instances if not provided
        if td is None:
            if self.generator is None:
                raise ValueError("Either provide td or set a generator for the environment")
            td = self.generator(batch_size or 1)

        # Move to device
        td = td.to(self.device)

        # Call problem-specific reset
        td = self._reset(td, batch_size)

        # Add common fields
        td["action_mask"] = self._get_action_mask(td)
        td["done"] = torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

        return td

    def step(self, td: TensorDict) -> TensorDict:
        """
        Public step with common postprocessing.

        This method handles the common step logic and delegates
        problem-specific state transition to _step().

        Args:
            td: TensorDict containing current state and action.

        Returns:
            TensorDict with updated state after action execution.
        """
        # Execute problem-specific step
        td = self._step(td)

        # Update common fields
        td["action_mask"] = self._get_action_mask(td)
        td["done"] = self._check_done(td)

        return td

    def get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Public method to compute rewards.

        Args:
            td: TensorDict containing the solution state.
            actions: Optional tensor of actions taken.

        Returns:
            Tensor of rewards.
        """
        return self._get_reward(td, actions)

    def to(self, device: Union[str, torch.device]) -> "RL4COEnvBase":
        """
        Move environment to specified device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        self.device = torch.device(device)
        return self

    @staticmethod
    def _make_spec(td: TensorDict) -> dict:
        """
        Create specification dict describing the observation and action spaces.

        This is useful for compatibility with RL libraries that expect
        explicit space definitions.

        Args:
            td: Sample TensorDict to infer shapes from.

        Returns:
            Dictionary describing observation and action spaces.
        """
        return {
            "observation_shape": dict(td.items()),
            "action_shape": td.get("action_mask", torch.tensor([])).shape,
        }

    def render(self, td: TensorDict, **kwargs: Any) -> Any:
        """
        Render the current state (optional).

        Override in subclasses for visualization support.

        Args:
            td: TensorDict containing current state.
            **kwargs: Rendering options.

        Returns:
            Rendered output (implementation-specific).
        """
        raise NotImplementedError(f"Rendering not implemented for {self.name}")

    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"


class ImprovementEnvBase(RL4COEnvBase):
    """
    Base environment for improvement-based methods.

    Improvement methods start with an initial solution and iteratively
    improve it through local search operations.
    """

    name: str = "improvement_base"

    @abstractmethod
    def _get_initial_solution(self, td: TensorDict) -> torch.Tensor:
        """
        Generate initial solution for improvement.

        Args:
            td: TensorDict containing the problem instance.

        Returns:
            Tensor representing the initial solution.
        """
        raise NotImplementedError

    def reset(self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None) -> TensorDict:
        """
        Reset with initial solution generation.

        Args:
            td: Optional TensorDict containing problem instance.
            batch_size: Batch size for generating new instances.

        Returns:
            TensorDict with initialized state including initial solution.
        """
        td = super().reset(td, batch_size)

        # Generate initial solution
        td["solution"] = self._get_initial_solution(td)

        return td
