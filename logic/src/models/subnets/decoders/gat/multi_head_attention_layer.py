"""Multi-Head Attention Layer for GAT Decoder.

This module implements a single layer of the graph attention mechanism, consisting
of multi-head attention followed by a feed-forward sublayer with skip connections
and normalization.

Attributes:
    MultiHeadAttentionLayer: Single transformer-style layer for GAT decoding.

Example:
    >>> from logic.src.models.subnets.decoders.gat.multi_head_attention_layer import MultiHeadAttentionLayer
    >>> layer = MultiHeadAttentionLayer(n_heads=8, embed_dim=128, feed_forward_hidden=512)
    >>> out = layer(query, node_embeddings, mask)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.decoders.common import FeedForwardSubLayer
from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)


class MultiHeadAttentionLayer(nn.Module):
    """Single layer of the Graph Attention Decoder.

    Contains Multi-Head Attention followed by a Feed-Forward block, both with
    skip connections and layer normalization.

    Attributes:
        att (SkipConnection): Multi-head attention sublayer with skip connection.
        norm1 (Normalization): Normalization after attention.
        ff (SkipConnection): Feed-forward sublayer with skip connection.
        norm2 (Normalization): Normalization after feed-forward.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the MultiHeadAttentionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for the FFN block.
            norm_config: Configuration for normalization layers.
            activation_config: Configuration for activation functions.
            kwargs: Additional keyword arguments.
        """
        super(MultiHeadAttentionLayer, self).__init__()

        if norm_config is None:
            norm_config = NormalizationConfig()
        if activation_config is None:
            activation_config = ActivationConfig()

        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(
            embed_dim,
            norm_config=norm_config,
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
            norm_config=norm_config,
        )

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            q: Query tensor.
            h: Key/Value (node) embeddings.
            mask: Attention mask.

        Returns:
            torch.Tensor: Normalized output embeddings.
        """
        h = self.att(q, h, mask)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)
