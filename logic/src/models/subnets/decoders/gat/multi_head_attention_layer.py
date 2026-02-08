"""Multi-Head Attention Layer for GAT Decoder."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)

from ..common import FeedForwardSubLayer


class MultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the Graph Attention Decoder.
    Contains Multi-Head Attention followed by a Feed-Forward block, with normalization.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs,
    ):
        """Initializes the MultiHeadAttentionLayer."""
        super(MultiHeadAttentionLayer, self).__init__()

        if norm_config is None:
            norm_config = NormalizationConfig()
        if activation_config is None:
            activation_config = ActivationConfig()

        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(
            embed_dim,
            norm_config.norm_type,
            norm_config.epsilon,
            norm_config.learn_affine,
            norm_config.track_stats,
            norm_config.momentum,
            norm_config.n_groups,
            norm_config.k_lrnorm,
        )
        self.ff = SkipConnection(
            FeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation_config=activation_config,
            )
        )
        self.norm2 = Normalization(
            embed_dim,
            norm_config.norm_type,
            norm_config.epsilon,
            norm_config.learn_affine,
            norm_config.track_stats,
            norm_config.momentum,
            norm_config.n_groups,
            norm_config.k_lrnorm,
        )

    def forward(self, q, h, mask):
        """Forward pass."""
        h = self.att(q, h, mask)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)
