"""
RL4CO Base Environment.
"""

from typing import Any, Optional, Union

import torch
from torchrl.envs import EnvBase

from .batch import BatchMixin
from .ops import OpsMixin


class RL4COEnvBase(BatchMixin, OpsMixin, EnvBase):
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

    def render(self, td: Any, **kwargs: Any) -> Any:
        """
        Render the current state (optional).
        """
        raise NotImplementedError(f"Rendering not implemented for {self.name}")

    def __repr__(self) -> str:
        """String representation of the environment."""
        return f"{self.__class__.__name__}(name={self.name}, device={self.device})"
