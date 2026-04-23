"""Feed-Forward network module.

This module provides the FeedForward layer, which implements a standard
position-wise multi-layer perceptron (MLP) block for spatial feature refinement.

Attributes:
    FeedForward: Standard Feed-Forward Neural Network (MLP) block.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.feed_forward import FeedForward
    >>> ffn = FeedForward(input_dim=128, output_dim=512)
    >>> x = torch.randn(1, 10, 128)
    >>> out = ffn(x)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class FeedForward(nn.Module):
    """Standard Feed-Forward Neural Network (MLP) block.

    Consists of a linear transformation layer, typically followed by an
    activation function (handled by the caller or container module).

    Attributes:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        linear (nn.Linear): Linear transformation layer.
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        """Initializes FeedForward.

        Args:
            input_dim: Dimensionality of the incoming features.
            output_dim: Dimensionality of the outgoing features.
            bias: Whether to add a learnable bias term to the transformation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the weights using a uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Applies the linear transformation to the input sequence.

        Args:
            input: Input feature tensor of shape (..., input_dim).
            mask: Optional tensor mask (currently ignored).

        Returns:
            torch.Tensor: Transformed tensor of shape (..., output_dim).
        """
        return self.linear(input)
