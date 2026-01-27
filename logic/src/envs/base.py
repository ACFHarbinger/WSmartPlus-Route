"""
Base environment class for combinatorial optimization problems.

This module provides the foundation for RL4CO-style environment abstraction,
enabling unified state management via TensorDict and problem-agnostic interfaces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Union, cast

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

    @property
    def batch_size(self) -> torch.Size:
        """Batch size of the environment."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: torch.Size) -> None:
        """Set the batch size of the environment."""
        # Check if value is a torch.Size object
        if not isinstance(value, torch.Size):
            if isinstance(value, int):
                value = torch.Size([value])
            else:
                value = torch.Size(value)

        try:
            # Try to let EnvBase handle it
            super(RL4COEnvBase, self.__class__).batch_size.fset(self, value)
        except (ValueError, RuntimeError):
            # Suppress spec re-indexing errors in 0.3.1
            self._batch_size = value

            def _safe_set_shape(s, shp):
                if s is None:
                    return
                try:
                    s.shape = shp
                except (ValueError, RuntimeError):
                    if hasattr(s, "items"):
                        for _, v in s.items():
                            _safe_set_shape(v, shp)

            # Manually sync spec shapes if EnvBase failed to do so
            if hasattr(self, "reward_spec"):
                _safe_set_shape(self.reward_spec, (*value, 1))
            if hasattr(self, "done_spec"):
                _safe_set_shape(self.done_spec, (*value, 1))

            # Sync container shapes
            for spec_name in ["observation_spec", "action_spec", "input_spec", "output_spec"]:
                try:
                    # Use getattr safely as these are often properties
                    spec = getattr(self, spec_name, None)
                    if spec is not None:
                        if hasattr(spec, "shape"):
                            try:
                                spec.shape = value
                            except (ValueError, RuntimeError):
                                pass
                        # Also sync internal done/terminated if they are in there
                        if hasattr(spec, "items"):
                            for k, v in spec.items():
                                if k in ["done", "terminated", "truncated", "reward"]:
                                    _safe_set_shape(v, (*value, 1))
                except (KeyError, AttributeError):
                    continue
        except Exception:
            # Fallback for older TorchRL
            self._batch_size = value

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
        # Default batch_size to empty Size if not provided
        if batch_size is None:
            batch_size = torch.Size([])
        else:
            if isinstance(batch_size, int):
                batch_size = torch.Size([batch_size])
            batch_size = torch.Size(batch_size)  # Ensure it's a torch.Size object

        # Filter kwargs for EnvBase
        env_base_kwargs = {
            "device": device,
            "batch_size": batch_size,
        }
        for k in list(kwargs.keys()):
            if k in ["run_type_checks", "allow_done_after_reset"]:
                env_base_kwargs[k] = kwargs.pop(k)

        super().__init__(**env_base_kwargs)

        # Manually set check_env_specs AFTER super init to avoid TypeError in 0.3.1
        self.check_env_specs = False

        # Manually set check_env_specs if provided (bypass strict checks in reset)
        self.check_env_specs = kwargs.get("check_env_specs", False)

        self.generator = generator
        self.generator_params = generator_params or {}
        self._kwargs = kwargs

    def _make_spec(self, generator: Optional[Any] = None) -> None:
        """
        Create environment specs (reward_spec, done_spec, etc.).

        Args:
            generator: Optional data generator.
        """
        from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec

        self.done_spec = DiscreteTensorSpec(n=2, shape=(*self.batch_size, 1), dtype=torch.bool, device=self.device)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1), device=self.device)

    def step(self, td: TensorDict) -> TensorDict:
        """
        Execute action and update state.
        Synchronizes environment batch size with input TensorDict.
        """
        # print(f"DEBUG: step() entering. td.batch_size={td.batch_size}, self.batch_size={self.batch_size}")
        self.batch_size = td.batch_size
        out = cast(TensorDict, super().step(td))
        # print(f"DEBUG: step() exiting. out['next']['done'].shape={out['next']['done'].shape}")
        return out

    def reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """
        Sync batch size and initialize state.
        """
        # Support both 'td' and 'tensordict' naming
        if td is None and "tensordict" in kwargs:
            td = kwargs.pop("tensordict")

        if td is not None:
            self.batch_size = td.batch_size

        return self._reset(td, **kwargs)

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[int] = None) -> TensorDict:
        """
        Initialize episode state from problem instance.
        """
        if td is None:
            if self.generator is None:
                raise ValueError("Either provide td or set a generator for the environment")
            td = self.generator(batch_size or self.batch_size)
        else:
            # TorchRL requires out-of-place updates for reset
            # So we clone the structure (and data if needed, but shallow copy of dict + tensor ref is usually enough if we don't modify tensors in-place in a way that affects original)
            # However, safe bet is to clone.
            td = td.clone()

        # Move to device
        td = td.to(self.device)

        # Call problem-specific reset (must be implemented by subclasses)
        td = self._reset_instance(td)

        # Add common fields
        td["action_mask"] = self._get_action_mask(td)
        td["i"] = torch.zeros(td.batch_size, dtype=torch.long, device=self.device)

        # Initialize done signals with [B, 1] shape
        td["done"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)
        if "terminated" in td.keys():
            td["terminated"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)
        if "truncated" in td.keys():
            td["truncated"] = torch.zeros((*td.batch_size, 1), dtype=torch.bool, device=self.device)

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
        done = self._check_done(td_next)

        # Ensure done shape matches spec [B, 1]
        if done.dim() == len(td_next.batch_size):
            done = done.unsqueeze(-1)
        elif done.dim() > len(td_next.batch_size) + 1:
            done = done.reshape((*td_next.batch_size, 1))

        td_next["done"] = done
        td_next["terminated"] = done.clone()
        td_next["truncated"] = torch.zeros_like(done)

        # Reward calculation
        reward = self._get_reward(td_next, td.get("action", None))

        # Ensure reward shape matches batch_size [B, 1] for consistency
        if reward.dim() == len(td_next.batch_size):
            reward = reward.unsqueeze(-1)
        elif reward.dim() > len(td_next.batch_size) + 1:
            reward = reward.reshape((*td_next.batch_size, 1))

        td_next["reward"] = reward

        return td_next

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Problem-specific state transition logic.

        Args:
            td: TensorDict containing the current state.

        Returns:
            TensorDict: The updated state (next state).
        """
        """
        Core state transition logic common to most routing problems.
        Updates visited mask, current node, and tour tracking.
        """
        action = td["action"]
        current = td.get("current_node", torch.zeros_like(action))

        # Robustly squeeze to [B]
        if current.dim() > 1:
            current = current.squeeze(-1)
        if action.dim() > 1:
            action = action.squeeze(-1)
        # Ensure they are at least 1D for slicing
        if current.dim() == 0:
            current = current.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        locs = td["locs"]

        # Compute distance traveled
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        next_loc = locs.gather(1, action[:, None, None].expand(-1, -1, 2)).squeeze(1)
        distance = torch.norm(next_loc - current_loc, dim=-1)

        # Update tour length
        td["tour_length"] = td.get("tour_length", torch.zeros_like(distance)) + distance

        # Update visited
        td["visited"] = td["visited"].scatter(1, action.unsqueeze(-1), True)

        # Update current node
        td["current_node"] = action.unsqueeze(-1)

        # Append to tour
        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

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
        Returns: [B] bool tensor
        """
        current_node = td.get("current_node", None)
        step_count = td.get("i", None)

        if current_node is None or step_count is None:
            return torch.zeros(td.batch_size, dtype=torch.bool, device=td.device)

        # Done if we're back at depot (node 0) after at least one step
        # Handle current_node being [B] or [B, 1]
        node = current_node.squeeze(-1) if current_node.dim() > 1 else current_node
        steps = step_count.squeeze(-1) if step_count.dim() > 1 else step_count
        done = (node == 0) & (steps > 0)
        try:
            return done.reshape(td.batch_size)
        except Exception:
            return done.flatten().reshape(td.batch_size)

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
