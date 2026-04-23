"""Transformer Graph Convolution Encoder.

Attributes:
    TransGraphConvEncoder: Encoder stack of MultiHeadAttentionLayers and GraphConvolutionLayers.

Example:
    >>> from logic.src.models.subnets.encoders.tgc import TransGraphConvEncoder
    >>> encoder = TransGraphConvEncoder(n_heads=8, embed_dim=128, n_layers=3)
"""

from typing import List, Optional

import torch
from torch import nn

from .conv_layer import GraphConvolutionLayer
from .mha_layer import TGCMultiHeadAttentionLayer


class TransGraphConvEncoder(nn.Module):
    """Encoder stack of MultiHeadAttentionLayers and GraphConvolutionLayers.

    Attributes:
        layers (nn.ModuleList): List of attention and convolution layers.
        dropout (nn.Dropout): Dropout layer applied to final embeddings.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
        epsilon_alpha: float = 1e-05,
        learn_affine: bool = True,
        track_stats: bool = False,
        momentum_beta: float = 0.1,
        locresp_k: float = 1.0,
        n_groups: int = 3,
        activation: str = "gelu",
        af_param: float = 1.0,
        threshold: float = 6.0,
        replacement_value: float = 6.0,
        n_params: int = 3,
        uniform_range: Optional[List[float]] = None,
        dropout_rate: float = 0.1,
        agg: str = "mean",
    ) -> None:
        """Initializes the TransGraphConvEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension size.
            n_layers: Number of attention layers.
            n_sublayers: Number of GCN layers. If None, no GCN layers are added.
            feed_forward_hidden: Hidden dimension size of feed-forward sublayers.
            normalization: Type of normalization to use.
            epsilon_alpha: Small value for numerical stability in normalization.
            learn_affine: Whether to learn affine parameters in normalization.
            track_stats: Whether to track running stats in normalization.
            momentum_beta: Momentum for running stats.
            locresp_k: K value for Local Response Normalization.
            n_groups: Number of groups for Group Normalization.
            activation: Activation function to use.
            af_param: Parameter for the activation function (e.g., gain).
            threshold: Threshold value for clipped activation functions.
            replacement_value: Value to replace when threshold is exceeded.
            n_params: Number of parameters for certain activations.
            uniform_range: Range for uniform initialization of parameters.
            dropout_rate: Dropout probability.
            agg: Aggregation method for graph convolution.
        """
        if uniform_range is None:
            uniform_range = [0.125, 1 / 3]
        super(TransGraphConvEncoder, self).__init__()
        layers = [
            TGCMultiHeadAttentionLayer(
                n_heads,
                embed_dim,
                feed_forward_hidden,
                normalization,
                epsilon_alpha,
                learn_affine,
                track_stats,
                momentum_beta,
                locresp_k,
                n_groups,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
            for _ in range(n_layers)
        ] + [
            GraphConvolutionLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
                normalization,
                epsilon_alpha,
                learn_affine,
                track_stats,
                momentum_beta,
                locresp_k,
                n_groups,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
            for _ in range(n_sublayers or 0)
        ]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        Args:
            x: Input embeddings of shape (batch_size, graph_size, embed_dim).
            edges: Adjacency/Mask matrix of shape (batch_size, graph_size, graph_size).

        Returns:
            torch.Tensor: Encoded embeddings of shape (batch_size, graph_size, embed_dim).
        """
        for layer in self.layers:
            x = layer(x, edges)
        return self.dropout(x)
