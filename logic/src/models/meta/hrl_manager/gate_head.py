"""Gate Head for Route Dispatch Decisions.

This module provides the `GateHead`, which determines whether to initiate
a routing cycle based on global occupancy and predicted urgency.

Attributes:
    GateHead: MLP-based binary categorical decision head.

Example:
    None
"""

from __future__ import annotations

import torch
from torch import nn


class GateHead(nn.Module):
    """MLP head for fleet dispatch gating.

    Produces categorical logits for the binary decision: dispatching vs skipping
    the routing cycle for the current period.

    Attributes:
        net (nn.Sequential): MLP layers (Linear -> ReLU -> Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the gate head.

        Args:
            input_dim: size of pooled global features.
            hidden_dim: width of the intermediate layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts dispatch logits.

        Args:
            x: global state representation [B, D].

        Returns:
            torch.Tensor: logits for [Skip, Dispatch] [B, 2].
        """
        return self.net(x)
