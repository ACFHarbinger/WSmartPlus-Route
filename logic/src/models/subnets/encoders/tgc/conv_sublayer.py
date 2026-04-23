"""Feed-Forward Convolution Sub-Layer for TGC.

Attributes:
    FFConvSubLayer: Feed-Forward Convolution Sub-Layer.

Example:
    >>> from logic.src.models.subnets.encoders.tgc.conv_sublayer import FFConvSubLayer
    >>> layer = FFConvSubLayer(embed_dim=128, feed_forward_hidden=512, agg="mean", ...)
"""

from typing import List, Optional

import torch
from torch import nn

from logic.src.models.subnets.modules import (
    ActivationFunction,
    FeedForward,
    GraphConvolution,
)


class FFConvSubLayer(nn.Module):
    """Feed-Forward Convolution Sub-Layer.

    Attributes:
        conv (GraphConvolution): Graph convolution module.
        af (ActivationFunction): Activation function module.
        ff (FeedForward): Feed-forward linear projection.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        agg: str,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        dist_range: List[float],
        bias: bool = True,
    ) -> None:
        """Initializes the FFConvSubLayer.

        Args:
            embed_dim: Dimension of the input and output embeddings.
            feed_forward_hidden: Hidden dimension size after convolution.
            agg: Aggregation method for graph convolution.
            activation: Name of the activation function.
            af_param: Parameter for the activation function.
            threshold: Threshold for clipped activations.
            replacement_value: Value to replace when threshold is exceeded.
            n_params: Number of parameters for certain activations.
            dist_range: Range for uniform initialization.
            bias: Whether to use bias in the final feed-forward layer.
        """
        super(FFConvSubLayer, self).__init__()
        self.conv = GraphConvolution(embed_dim, feed_forward_hidden, agg)
        self.af = ActivationFunction(activation, af_param, threshold, replacement_value, n_params, dist_range)
        self.ff = FeedForward(feed_forward_hidden, embed_dim, bias=bias)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the convolution sublayer.

        Args:
            h: Input embeddings of shape (batch_size, graph_size, embed_dim).
            mask: Optional adjacency/mask matrix of shape (batch_size, graph_size, graph_size).

        Returns:
            torch.Tensor: Processed embeddings of shape (batch_size, graph_size, embed_dim).
        """
        h = self.conv(h, mask)
        h = self.af(h)
        return self.ff(h)
