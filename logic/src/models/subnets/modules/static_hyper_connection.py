"""Static Hyper-Network connections.

This module provides the StaticHyperConnection layer, which uses fixed weight
expansion to mix information across multiple parallel streams.

Attributes:
    StaticHyperConnection: Hyper-connection layer with fixed weight expansion.

Example:
    >>> import torch
    >>> from torch import nn
    >>> from logic.src.models.subnets.modules.static_hyper_connection import StaticHyperConnection
    >>> module = nn.Linear(128, 128)
    >>> model = StaticHyperConnection(module, hyper_dim=128, expansion_rate=4)
    >>> h = torch.randn(1, 10, 128, 4)
    >>> out = model(h)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class StaticHyperConnection(nn.Module):
    """Hyper-connection with static width/depth expansion.

    Implements a simplified multi-stream information mixer where transformation
    weights are fixed learnable parameters instead of dynamic context-dependent
    predictions.

    Attributes:
        module (nn.Module): The transformation sub-module (e.g., Attention, MLP).
        n (int): Width/depth expansion rate (number of parallel streams).
        width_mixer (nn.Parameter): Static weights for mixing existing streams.
        input_mixer (nn.Parameter): Static weights for collapsing streams.
        depth_mixer (nn.Parameter): Static weights for broadcasting sub-layer output.
    """

    def __init__(self, module: nn.Module, hyper_dim: int, expansion_rate: int = 4) -> None:
        """Initializes StaticHyperConnection.

        Args:
            module: The neural network module to wrap.
            hyper_dim: Hyper-network dimension (base embedding size).
            expansion_rate: Number of parallel information streams.
        """
        super().__init__()
        self.module = module
        self.n = expansion_rate

        # Initialize slightly off-identity to preserve gradient flow at start
        self.width_mixer = nn.Parameter(torch.eye(self.n) + torch.randn(self.n, self.n) * 0.01)
        self.input_mixer = nn.Parameter(torch.randn(self.n, 1) * 0.01)
        self.depth_mixer = nn.Parameter(torch.randn(1, self.n) * 0.01)

    def forward(self, H: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Applies static mixing to input streams.

        Args:
            H: Stream tensor of shape (batch, seq, dim, n).
            args: Positional arguments for the wrapped `module`.
            kwargs: Keyword arguments for the wrapped `module`.

        Returns:
            torch.Tensor: Updated stream tensor of shape (batch, seq, dim, n).
        """
        # Collapse streams for the sub-layer (A_m)
        h_in = torch.matmul(H, self.input_mixer).squeeze(-1)

        # Apply Sub-layer (Attention, MLP, etc.)
        y = self.module(h_in, *args, **kwargs)  # (B, S, D)

        # Update Hyper Matrix
        # Width: Mix existing streams (H x A_r)
        term_width = torch.matmul(H, self.width_mixer)

        # Depth: Broadcast new info (y x B)
        term_depth = torch.matmul(y.unsqueeze(-1), self.depth_mixer)

        return term_width + term_depth
