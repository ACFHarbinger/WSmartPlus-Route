"""Feed-Forward Sub-Layer for Encoder Architectures.

Attributes:
    EncoderFeedForwardSubLayer: Reusable Feed-Forward Sub-Layer for Encoder Architectures.

Example:
    >>> from logic.src.models.subnets.encoders.common.feed_forward_sublayer import EncoderFeedForwardSubLayer
    >>> layer = EncoderFeedForwardSubLayer(embed_dim=128, feed_forward_hidden=512)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class EncoderFeedForwardSubLayer(nn.Module):
    """Reusable Feed-Forward Sub-Layer for Encoder Architectures.

    Implements the standard transformer feed-forward pattern:
    1. Linear expansion: embed_dim → feed_forward_hidden
    2. Non-linear activation function
    3. Linear projection: feed_forward_hidden → embed_dim

    This component is used across multiple encoder types to provide consistent
    feed-forward transformation with configurable activation.

    Attributes:
        sub_layers (nn.Sequential): Sequential module containing the FFN layers.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the EncoderFeedForwardSubLayer.

        Args:
            embed_dim: Embedding dimension (input and output size).
            feed_forward_hidden: Hidden dimension for the feed-forward network.
                If 0, creates an identity mapping (embed_dim → embed_dim).
            activation_config: Activation configuration.
            bias: Whether to include bias terms in linear layers.
            kwargs: Additional keyword arguments.
        """
        super(EncoderFeedForwardSubLayer, self).__init__()

        if activation_config is None:
            activation_config = ActivationConfig()

        if feed_forward_hidden > 0:
            self.sub_layers = nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation_config=activation_config,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        else:
            self.sub_layers = nn.Sequential(FeedForward(embed_dim, embed_dim, bias=bias))

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            h: Input tensor of shape (batch, seq_len, embed_dim)
                or (batch, seq_len, embed_dim, n) for hyper-connections.
            mask: Optional mask tensor (unused, kept for signature compatibility).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        return self.sub_layers(h)
