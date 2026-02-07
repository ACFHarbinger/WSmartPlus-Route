"""Abstract base class for environment context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class EnvContext(nn.Module, ABC):
    """
    Abstract base class for environment context.
    Provides context for the current step in the environment.
    """

    def __init__(self, embed_dim: int, step_context_dim: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self._step_context_dim = step_context_dim

    @property
    def step_context_dim(self) -> int:
        return self._step_context_dim

    @abstractmethod
    def forward(self, state: Any) -> torch.Tensor:
        """Get context embedding for the current state."""
        pass
