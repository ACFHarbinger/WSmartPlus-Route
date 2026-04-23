"""Critic Head for State Value Estimation.

This module provides the `CriticHead`, which estimates the expected cumulative
discounted reward from a given global problem state.

Attributes:
    CriticHead: MLP-based value function estimator.
"""

from __future__ import annotations

import torch
from torch import nn


class CriticHead(nn.Module):
    """MLP head for value function estimation in PPO.

    Projects the combined global problem features to a single scalar value.

    Attributes:
        net (nn.Sequential): MLP layers (Linear -> ReLU -> Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the critic head.

        Args:
            input_dim: size of the incoming feature vector (pooled spatial + global).
            hidden_dim: width of the intermediate representation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimates the state value.

        Args:
            x: current global feature representation [B, D].

        Returns:
            torch.Tensor: predicted value scalars [B, 1].
        """
        return self.net(x)
