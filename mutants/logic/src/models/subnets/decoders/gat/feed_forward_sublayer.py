"""Feed-Forward Sub-layer for GAT Decoder."""

from __future__ import annotations

import torch.nn as nn
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class FeedForwardSubLayer(nn.Module):
    """
    Sub-layer containing a Feed-Forward Network and activation.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
    ):
        """Initializes the FeedForwardSubLayer."""
        super(FeedForwardSubLayer, self).__init__()
        self.sub_layers = (
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    dist_range,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h, mask=None):
        """Forward pass."""
        return self.sub_layers(h)
