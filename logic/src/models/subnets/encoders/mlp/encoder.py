"""MLP Encoder."""

from __future__ import annotations

import torch.nn as nn

from .mlp_layer import MLPLayer


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with ReLU activation, independent of graph structure.
    """

    def __init__(
        self,
        n_layers,
        feed_forward_hidden,
        norm="layer",
        learn_affine=True,
        track_norm=False,
        *args,
        **kwargs,
    ):
        """
        Initializes the MLP Encoder.
        """
        super(MLPEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MLPLayer(feed_forward_hidden, norm, learn_affine, track_norm) for _ in range(n_layers)]
        )

    def forward(self, x, graph=None):
        """
        Forward pass.
        """
        for layer in self.layers:
            x = layer(x)

        return x
