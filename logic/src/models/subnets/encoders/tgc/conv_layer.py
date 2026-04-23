"""Graph Convolution Layer for TGC.

Attributes:
    GraphConvolutionLayer: Graph Convolution Layer with Normalization.

Example:
    >>> from logic.src.models.subnets.encoders.tgc.conv_layer import GraphConvolutionLayer
    >>> layer = GraphConvolutionLayer(embed_dim=128, feed_forward_hidden=512, agg="mean", ...)
"""

from typing import List

import torch
from torch import nn

from logic.src.models.subnets.modules import Normalization, SkipConnection

from .conv_sublayer import FFConvSubLayer


class GraphConvolutionLayer(nn.Module):
    """Graph Convolution Layer with Normalization.

    Attributes:
        ff_conv (SkipConnection): Convolution sublayer with skip connection.
        norm (Normalization): Normalization layer.
    """

    def __init__(
        self,
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
        """Initializes the GraphConvolutionLayer.

        Args:
            embed_dim: Embedding dimension size.
            feed_forward_hidden: Hidden dimension size of feed-forward sublayer.
            agg: Aggregation method for graph convolution.
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
        super(GraphConvolutionLayer, self).__init__()
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
        self.norm = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convolution layer.

        Args:
            h: Input embeddings of shape (batch_size, graph_size, embed_dim).
            mask: Adjacency/Mask matrix of shape (batch_size, graph_size, graph_size).

        Returns:
            torch.Tensor: Normalized embeddings after graph convolution.
        """
        h = self.ff_conv(h, mask=mask)
        return self.norm(h)
