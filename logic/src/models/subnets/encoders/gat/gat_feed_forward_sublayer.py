"""Feed-Forward Sub-Layer for GAT Encoder.

Attributes:
    GATFeedForwardSubLayer: Feed-Forward Sub-Layer for GAT Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gat.gat_feed_forward_sublayer import GATFeedForwardSubLayer
    >>> layer = GATFeedForwardSubLayer(embed_dim=128, feed_forward_hidden=512)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class GATFeedForwardSubLayer(nn.Module):
    """Feed-Forward Sub-Layer for GAT Encoder.

    Processes embeddings through a expansion layer, an activation function,
    and a final projection layer.

    Attributes:
        sub_layers (nn.Module): Sequential container for sub-layers (FFN expansion,
            activation, FFN projection) or a single FeedForward layer for identity.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the GATFeedForwardSubLayer.

        Args:
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for feed-forward layers.
                If 0, creates a single identity linear projection.
            activation_config: Activation function configuration.
            bias: Whether to use bias in linear layers.
            kwargs: Additional keyword arguments.
        """
        super(GATFeedForwardSubLayer, self).__init__()

        if activation_config is None:
            activation_config = ActivationConfig()

        if feed_forward_hidden > 0:
            self.sub_layers = nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(activation_config=activation_config),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        else:
            self.sub_layers = FeedForward(embed_dim, embed_dim, bias=bias)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the GAT feed-forward sub-layer.

        Args:
            h: Input tensor of shape (batch, nodes, embed_dim).
            mask: Optional mask tensor (unused).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        return self.sub_layers(h)
