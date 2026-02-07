"""Abstract base class for context embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class ContextEmbedder(nn.Module, ABC):
    """
    Abstract base class for problem-specific context embeddings.
    Responsible for initializing node embeddings and determining step context dimensions.
    """

    def __init__(self, embed_dim: int, node_dim: int, temporal_horizon: int):
        """
        Initialize the ContextEmbedder.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon
        self.init_embed = None

    @abstractmethod
    def init_node_embeddings(self, nodes: dict[str, Any]) -> torch.Tensor:
        """Initialize node embeddings from input data."""
        pass

    @property
    @abstractmethod
    def step_context_dim(self) -> int:
        """Get the dimension of the step context."""
        pass

    def forward(self, input: dict[str, Any]) -> torch.Tensor:
        return self.init_node_embeddings(input)
