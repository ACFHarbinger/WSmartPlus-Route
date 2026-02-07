"""Attention layer for MDAM Encoder."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from logic.src.models.subnets.modules.multi_head_attention import MultiHeadAttention
from logic.src.models.subnets.modules.normalization import Normalization
from logic.src.models.subnets.modules.skip_connection import SkipConnection


class MultiHeadAttentionLayer(nn.Module):
    """
    Single transformer layer with MHA + FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
    ) -> None:
        """
        Initialize attention layer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            feed_forward_hidden: Hidden dimension for FFN.
            normalization: Type of normalization ('batch', 'layer', 'instance').
        """
        super().__init__()

        self.attention = MultiHeadAttention(
            n_heads=num_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim,
        )

        self.norm1 = Normalization(embed_dim, normalization)

        self.ffn = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )

        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        """
        # Self-attention with residual
        h = x + self.attention(x, x, mask)
        h = self.norm1(h)

        # FFN with residual (handled by SkipConnection)
        h = self.ffn(h)
        h = self.norm2(h)

        return h
