"""env.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import env
"""

from typing import Any, Optional, Protocol

import torch
from tensordict import TensorDict


class IEnv(Protocol):
    """
    Protocol for RL Environments.
    Matches RL4COEnvBase interface.
    """

    name: str
    device: torch.device
    generator: Optional[Any] = None

    def reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        """Reset environment."""
        ...

    def step(self, td: TensorDict) -> TensorDict:
        """Step environment."""
        ...

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> TensorDict:
        """Calculate reward."""
        ...

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Get valid action mask."""
        ...
