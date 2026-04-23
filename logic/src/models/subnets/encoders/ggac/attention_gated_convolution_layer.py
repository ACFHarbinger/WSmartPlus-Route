"""Attention Gated Convolution Layer for GGAC Encoder.

Attributes:
    AttentionGatedConvolutionLayer: Combined Layer with MHA, Gated GCN, and Feed-Forward Network.

Example:
    >>> from logic.src.models.subnets.encoders.ggac.attention_gated_convolution_layer import AttentionGatedConvolutionLayer
    >>> layer = AttentionGatedConvolutionLayer(n_heads=8, embed_dim=128, ...)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from logic.src.models.subnets.modules import (
    ActivationFunction,
    FeedForward,
    GatedGraphConvolution,
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)


class AttentionGatedConvolutionLayer(nn.Module):
    """Combined Layer with MHA, Gated GCN, and Feed-Forward Network.

    Attributes:
        att (SkipConnection): Multi-head attention sublayer.
        norm1 (Normalization): First normalization layer.
        gated_gcn (GatedGraphConvolution): Gated graph convolution sublayer.
        norm2 (Normalization): Second normalization layer.
        ff (SkipConnection): Feed-forward sublayer.
        norm3 (Normalization): Third normalization layer.
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
        gated: bool = True,
        agg: str = "sum",
        bias: bool = True,
    ) -> None:
        """Initializes the AttentionGatedConvolutionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for feed-forward layers.
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
            gated: Whether to use gated convolutions.
            agg: Aggregation method for graph convolutions.
            bias: Whether to use bias in linear layers.
        """
        super(AttentionGatedConvolutionLayer, self).__init__()

        # 1. Multi-Head Attention (Global Context)
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

        # 2. Gated Graph Convolution (Local Structure + Edge Updates)
        self.gated_gcn = GatedGraphConvolution(
            hidden_dim=embed_dim,
            aggregation=agg,
            norm=normalization,
            activation=activation,
            learn_affine=learn_affine,
            gated=gated,
            bias=bias,
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

        # 3. Feed Forward (Node-wise processing)
        self.ff = (
            SkipConnection(
                nn.Sequential(
                    FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                    ActivationFunction(
                        activation,
                        af_param,
                        threshold,
                        replacement_value,
                        n_params,
                        uniform_range,
                    ),
                    FeedForward(feed_forward_hidden, embed_dim, bias=bias),
                )
            )
            if feed_forward_hidden > 0
            else SkipConnection(FeedForward(embed_dim, embed_dim))
        )

        self.norm3 = Normalization(
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
        e: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Node input tensor of shape (batch, num_nodes, embed_dim).
            e: Edge input tensor of shape (batch, num_nodes, num_nodes, embed_dim).
            mask: Optional attention mask tensor of shape (batch, num_nodes, num_nodes).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated node and edge features.
        """
        # 1. MHA
        h = self.att(h, mask=mask)
        h = self.norm1(h)

        # 2. Gated GCN
        h_in = h
        e_in = e

        h_gcn, e_gcn = self.gated_gcn(h, e, mask)

        h = h_in + h_gcn  # Residual for Node
        e = e_in + e_gcn  # Residual for Edge

        h = self.norm2(h)

        # 3. FF
        h = self.ff(h)
        h = self.norm3(h)

        return h, e
