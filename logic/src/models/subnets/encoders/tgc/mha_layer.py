"""Multi-Head Attention Layer for TGC.

Attributes:
    TGCMultiHeadAttentionLayer: Multi-Head Attention Layer with Normalization and Feed-Forward.

Example:
    >>> from logic.src.models.subnets.encoders.tgc.mha_layer import TGCMultiHeadAttentionLayer
    >>> layer = TGCMultiHeadAttentionLayer(n_heads=8, embed_dim=128, feed_forward_hidden=512, ...)
"""

from typing import List, Optional

import torch
from torch import nn

from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)

from .ff_sublayer import TGCFeedForwardSubLayer


class TGCMultiHeadAttentionLayer(nn.Module):
    """Multi-Head Attention Layer with Normalization and Feed-Forward.

    Attributes:
        att (SkipConnection): Attention sublayer with skip connection.
        norm1 (Normalization): First normalization layer.
        ff (SkipConnection): Feed-forward sublayer with skip connection.
        norm2 (Normalization): Second normalization layer.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str,
        epsilon_alpha: float,
        learn_affine: bool,
        track_stats: bool,
        mbeta: float,
        lr_k: float,
        n_groups: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        uniform_range: List[float],
    ) -> None:
        """Initializes the TGCMultiHeadAttentionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension size.
            feed_forward_hidden: Hidden dimension size of feed-forward sublayer.
            normalization: Type of normalization to use.
            epsilon_alpha: Stability constant for normalization.
            learn_affine: Whether to learn affine parameters.
            track_stats: Whether to track running stats in normalization.
            mbeta: Momentum for normalization stats.
            lr_k: K value for Local Response Normalization.
            n_groups: Number of groups for Group Normalization.
            activation: Activation function name.
            af_param: Activation function parameter.
            threshold: Threshold for clipped activations.
            replacement_value: Value to replace when threshold is exceeded.
            n_params: Number of parameters for certain activations.
            uniform_range: Range for uniform initialization.
        """
        super(TGCMultiHeadAttentionLayer, self).__init__()
        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )
        self.ff = SkipConnection(
            TGCFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
        )
        self.norm2 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with skip connections and normalization.

        Args:
            h: Input tensor of shape (batch_size, graph_size, embed_dim).
            mask: Optional attention mask of shape (batch_size, graph_size, graph_size).

        Returns:
            torch.Tensor: Normalized embeddings after attention and feed-forward.
        """
        h = self.att(h, mask=mask)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)
