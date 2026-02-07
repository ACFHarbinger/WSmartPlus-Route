"""
Transductive Model Base Classes.

This module provides base classes for transductive (search-time) methods.
Transductive methods adapt or search specifically on a test instance,
often starting from a pre-trained constructive or improvement policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict


class TransductiveModel(nn.Module, ABC):
    """
    Base class for transductive methods.

    Transductive models perform search or adaptation at inference time.
    Common examples include Active Search (Bello et al.), EAS (Hottung et al.),
    and other instance-specific fine-tuning methods.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[nn.Module] = None,
        **kwargs,
    ):
        """Initialize TransductiveModel."""
        super().__init__()
        self.env = env
        self.policy = policy

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute search or adaptation on the given instances.

        Args:
            td: TensorDict containing problem instance(s).

        Returns:
            Dictionary containing final results (reward, actions, etc.).
        """
        # This will typically involve an optimization loop
        raise NotImplementedError
