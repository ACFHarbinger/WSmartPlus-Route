"""
Base environment class for combinatorial optimization problems.

This module provides the foundation for RL4CO-style environment abstraction,
enabling unified state management via TensorDict and problem-agnostic interfaces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase


class RL4COEnvBase(EnvBase):
    """
    Base environment class for combinatorial optimization problems.

    This class provides a unified interface for problem environments following
    the torchrl/RL4CO architecture pattern. All problem-specific environments should
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
        batch_size: Optional[Union[torch.Size, int]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the base environment.

        Args:
            generator: Data generator instance for creating problem instances.
            generator_params: Parameters to pass to the generator if not provided.
            device: Device to place tensors on.
            batch_size: Batch size for the environment.
            **kwargs: Additional keyword arguments.
        """
        if batch_size is None:
            batch_size = torch.Size([])
        elif isinstance(batch_size, int):
            batch_size = torch.Size([batch_size])

        super().__init__(device=device, batch_size=batch_size)
        self.generator = generator
        self.generator_params = generator_params or {}
        self._kwargs = kwargs

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None) -> TensorDict:
        """
        Initialize episode state from problem instance.
        """
        if td is None:
            if self.generator is None:
                raise ValueError("Either provide td or set a generator for the environment")
            td = self.generator(batch_size or self.batch_size)

        # Move to device
        td = td.to(self.device)

        # Call problem-specific reset (must be implemented by subclasses)
        td = self._reset_instance(td)

        # Add common fields
        td["action_mask"] = self._get_action_mask(td)
        td["i"] = torch.zeros((*td.batch_size, 1), dtype=torch.long, device=self.device)

        return td

    @abstractmethod
    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Problem-specific instance initialization."""
        raise NotImplementedError

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Execute action and return new state as 'next' entry.
        """
        # Copy and clean to avoid cycles
        td_next = td.copy()
        for key in ["next", "reward", "done"]:
            if key in td_next.keys():
                del td_next[key]

        # Execute problem-specific step
        td_next = self._step_instance(td_next)

        # Update common fields in the next state
        td_next["i"] = td["i"] + 1
        td_next["action_mask"] = self._get_action_mask(td_next)
        td_next["done"] = self._check_done(td_next)

        # Reward is usually computed at the end in CO, but can be step-wise
        # TorchRL expects 'reward' and 'done' in the output of _step
        td_next["reward"] = self._get_reward(td_next, td.get("action", None))

        return td_next

    @abstractmethod
    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Problem-specific state transition."""
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward."""
        raise NotImplementedError

    @abstractmethod
    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Return mask of valid actions."""
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int]):
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """
        Check if episodes are complete.
        """
        current_node = td.get("current_node", None)
        step_count = td.get("i", None)

        if current_node is None or step_count is None:
            return torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

        # Done if we're back at depot (node 0) after at least one step
        return (current_node.squeeze(-1) == 0) & (step_count.squeeze(-1) > 0)

    def get_reward(self, td: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Public method to compute rewards.
        """
        return self._get_reward(td, actions)

    def render(self, td: TensorDict, **kwargs: Any) -> Any:
        """
        Render the current state (optional).
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
        """
        raise NotImplementedError

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Reset with initial solution generation.
        """
        # Generate initial solution
        td["solution"] = self._get_initial_solution(td)

        return td
