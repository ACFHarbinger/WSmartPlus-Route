"""
Dynamic embeddings for step-dependent updates.

Aligns with rl4co's DynamicEmbedding pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


class StaticEmbedding(nn.Module):
    """Static embedding: No dynamic updates (glimpse_key==glimpse_val==logit_key==0)."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, td: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return 0, 0, 0


class DynamicEmbedding(nn.Module):
    """Base class for dynamic embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(embed_dim, 3 * embed_dim)

    def forward(self, td: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dynamic updates for glimpse K, V and logit K.
        Returns tuple of 3 tensors or 0s.
        """
        # Example implementation (placeholder)
        # subclasses should implement actual logic
        return 0, 0, 0
