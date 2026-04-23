"""Attention Convolution Layer for GAC Encoder.

Attributes:
    AttentionConvolutionLayer: Combined Attention and Convolution Layer.

Example:
    >>> from logic.src.models.subnets.encoders.gac.attention_convolution_layer import AttentionConvolutionLayer
    >>> layer = AttentionConvolutionLayer(n_heads=8, embed_dim=128, feed_forward_hidden=512, ...)
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn

from logic.src.models.subnets.modules import (
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)

from .ff_conv_sublayer import FFConvSubLayer


class AttentionConvolutionLayer(nn.Module):
    """Combined Attention and Convolution Layer.

    Attributes:
        att (SkipConnection): Attention sublayer with skip connection.
        norm1 (Normalization): First normalization layer.
        ff_conv (SkipConnection): Convolution sublayer with skip connection.
        norm2 (Normalization): Second normalization layer.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        agg: str,
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
        """Initializes the AttentionConvolutionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for feed-forward layers.
            agg: Aggregation method for graph convolutions.
            normalization: Type of normalization to use.
            epsilon_alpha: Stability constant for normalization.
            learn_affine: Whether to learn affine parameters.
            track_stats: Whether to track running stats in normalization.
            mbeta: Momentum for normalization stats.
            lr_k: Local response normalization parameter.
            n_groups: Number of groups for group normalization.
            activation: Activation function name.
            af_param: Activation function parameter.
            threshold: Threshold for clipped activations.
            replacement_value: Value to replace when threshold is exceeded.
            n_params: Number of parameters for certain activations.
            uniform_range: Range for uniform initialization.
        """
        super(AttentionConvolutionLayer, self).__init__()
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
        self.ff_conv = SkipConnection(
            FFConvSubLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
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

    def forward(
        self,
        h: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Input tensor of shape (batch_size, graph_size, embed_dim).
            edges: Optional adjacency/edge matrix of shape (batch_size, graph_size, graph_size).
            mask: Optional attention mask of shape (batch_size, graph_size, graph_size).

        Returns:
            torch.Tensor: Normalized embeddings after attention and convolution.
        """
        h = self.att(h, mask=mask)
        h = self.norm1(h)
        # Use edges if provided (typical for graph conv), otherwise fallback to mask
        adj = edges if edges is not None else mask
        h = self.ff_conv(h, mask=adj)
        return self.norm2(h)
