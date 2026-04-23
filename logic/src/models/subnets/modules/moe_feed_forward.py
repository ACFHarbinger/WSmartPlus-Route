"""MoE Feed Forward Network implementation.

This module provides the MoEFeedForward layer, which replaces standard
feed-forward sub-layers with a Mixture-of-Experts (MoE) block. Each expert
is a full FFN (Linear -> Activation -> Linear) and routing is handled via
a sparse gating network.

Attributes:
    MoEFeedForward: Sub-layer replacing standard FeedForward with Mixture of Experts.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward
    >>> ffn = MoEFeedForward(
    ...     embed_dim=128,
    ...     feed_forward_hidden=512,
    ...     activation="ReLU",
    ...     af_param=None,
    ...     threshold=0.0,
    ...     replacement_value=0.0,
    ...     n_params=0,
    ...     dist_range=None
    ... )
    >>> x = torch.randn(1, 10, 128)
    >>> out = ffn(x)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .activation_function import ActivationFunction
from .feed_forward import FeedForward
from .moe_layer import MoE


class MoEFeedForward(nn.Module):
    """Sub-layer replacing standard FeedForward with Mixture of Experts.

    Efficiently scales model capacity by selecting only `k` out of `num_experts`
    for each token in the input sequence. This allows for very large hidden
    dimensions with constant computation costs.

    Attributes:
        moe (MoE): The sparse gating and expert execution controller.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation: str,
        af_param: Optional[float],
        threshold: float,
        replacement_value: float,
        n_params: int,
        dist_range: Optional[Tuple[float, float]],
        bias: bool = True,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
    ) -> None:
        """Initializes MoEFeedForward.

        Args:
            embed_dim: Input and output dimensionality.
            feed_forward_hidden: Hidden layer width within each expert's FFN.
            activation: String identifier for the activation function.
            af_param: Scaling parameter for specific activation types.
            threshold: Activation pruning threshold.
            replacement_value: Constant substituted when thresholding triggers.
            n_params: Number of learnable parameters in the activation function.
            dist_range: Randomization range for parameterized activations.
            bias: Whether to use bias in internal linear layers.
            num_experts: Total pool of expert FFN blocks.
            k: Number of experts activated for each feature vector.
            noisy_gating: Whether to use stochastic noise for better expert load balancing.
        """
        super().__init__()

        # Create a list of experts.
        # Each expert is a standard FFN block: Linear -> Act -> Linear
        experts = nn.ModuleList(
            [
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
                for _ in range(num_experts)
            ]
        )

        # Initialize MoE logic
        self.moe = MoE(
            input_size=embed_dim,
            output_size=embed_dim,
            experts=experts,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies adaptive expert routing and computation.

        Args:
            h: Input state tensor of shape (batch, nodes, embed_dim).
            mask: Optional attention mask (unused by internal MoE implementation).

        Returns:
            torch.Tensor: Transformed states of shape (batch, nodes, embed_dim).
        """
        # MoE logic handles the dispatching and combination
        return self.moe(h)
