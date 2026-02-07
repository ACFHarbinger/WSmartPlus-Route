"""Feed-Forward Sub-Layer for GAT Encoder."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class GATFeedForwardSubLayer(nn.Module):
    """
    Sub-layer containing a Feed-Forward Network and activation.
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
        """Initializes the GATFeedForwardSubLayer."""
        super(GATFeedForwardSubLayer, self).__init__()
        self.sub_layers = (
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    (dist_range[0], dist_range[1])
                    if dist_range is not None
                    and isinstance(dist_range, (list, tuple, np.ndarray))
                    and len(dist_range) >= 2
                    else None,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        return self.sub_layers(h)
