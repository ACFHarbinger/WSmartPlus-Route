"""Feed-Forward Sub-Layer for TGC.

Attributes:
    TGCFeedForwardSubLayer: Feed-Forward Sub-Layer with activation.

Example:
    >>> from logic.src.models.subnets.encoders.tgc.ff_sublayer import TGCFeedForwardSubLayer
    >>> layer = TGCFeedForwardSubLayer(embed_dim=128, feed_forward_hidden=512, activation="gelu", ...)
"""

from typing import List, Optional

import torch
from torch import nn

from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class TGCFeedForwardSubLayer(nn.Module):
    """Feed-Forward Sub-Layer with activation.

    Attributes:
        sub_layers (nn.Module): Sequential module containing feed-forward and activation layers.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        dist_range: List[float],
        bias: bool = True,
    ) -> None:
        """Initializes the TGCFeedForwardSubLayer.

        Args:
            embed_dim: Dimension of the input and output embeddings.
            feed_forward_hidden: Dimension of the hidden layer.
            activation: Name of the activation function.
            af_param: Parameter for the activation function.
            threshold: Threshold for clipped activations.
            replacement_value: Value to replace when threshold is exceeded.
            n_params: Number of parameters for certain activations.
            dist_range: Range for uniform initialization.
            bias: Whether to use bias in feed-forward layers.
        """
        super(TGCFeedForwardSubLayer, self).__init__()
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

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the feed-forward sublayer.

        Args:
            h: Input embeddings of shape (batch_size, graph_size, embed_dim).
            mask: Unused mask argument (kept for signature compatibility).

        Returns:
            torch.Tensor: Processed embeddings of shape (batch_size, graph_size, embed_dim).
        """
        return self.sub_layers(h)
