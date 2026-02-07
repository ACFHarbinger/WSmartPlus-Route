"""Graph Attention Convolution Encoder."""

from __future__ import annotations

import torch.nn as nn

from .attention_convolution_layer import AttentionConvolutionLayer


class GraphAttConvEncoder(nn.Module):
    """
    Encoder using a stack of AttentionConvolutionLayers.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        n_sublayers=None,
        feed_forward_hidden=512,
        normalization="batch",
        epsilon_alpha=1e-05,
        learn_affine=True,
        track_stats=False,
        momentum_beta=0.1,
        locresp_k=1.0,
        n_groups=3,
        activation="gelu",
        af_param=1.0,
        threshold=6.0,
        replacement_value=6.0,
        n_params=3,
        uniform_range=[0.125, 1 / 3],
        dropout_rate=0.1,
        aggregate: str = "sum",
    ):
        """Initializes the GraphAttConvEncoder."""
        super(GraphAttConvEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                AttentionConvolutionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    aggregate,
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
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges):
        """Forward pass."""
        for layer in self.layers:
            x = layer(x, edges)
        return self.dropout(x)
