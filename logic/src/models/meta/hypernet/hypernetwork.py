"""HyperNetwork for Adaptive Cost Weight Generation.

This module implements a HyperNetwork that dynamically generates reward/cost
weights for Reinforcement Learning agents based on temporal context and
current performance metrics.

Attributes:
    HyperNetwork: Meta-model for time-variant weight adjustment.
"""

from __future__ import annotations

import torch
from torch import nn

from logic.src.models.subnets.modules import ActivationFunction, Normalization


class HyperNetwork(nn.Module):
    """Adaptive Weight-Generating HyperNetwork.

    Takes environmental temporal indices (day of year) and performance metrics (e.g.,
    current fleet utilization) and projects them into a set of adaptive weights
    used to scale different objective components in a composite reward function.

    Attributes:
        input_dim (int): dimension of the incoming metric vector.
        output_dim (int): count of adjustable reward weights generated.
        n_days (int): seasonal wrapping period (e.g., 365 for annual).
        time_embedding (nn.Embedding): latent temporal representation.
        layers (nn.Sequential): MLP core for feature fusion.
        activation (nn.Module): ensures positive scalar outputs (Softplus).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_days: int = 365,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        normalization: str = "layer",
        activation: str = "relu",
        learn_affine: bool = True,
        bias: bool = True,
    ) -> None:
        """Initializes the HyperNetwork.

        Args:
            input_dim: static/dynamic metrics size.
            output_dim: expected number of reward weights.
            n_days: cyclical period for time embeddings.
            embed_dim: size of the latent time vector.
            hidden_dim: width of the fusion layers.
            normalization: type of normalization ('layer', 'batch', etc.).
            activation: non-linear function type ('relu', 'silu', etc.).
            learn_affine: whether to train norm parameters.
            bias: whether to use bias terms in linear layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_days = n_days
        self.time_embedding = nn.Embedding(n_days, embed_dim)

        combined_dim = input_dim + embed_dim

        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, output_dim, bias=bias),
        )

        self.activation = nn.Softplus()
        self.init_weights()

    def init_weights(self) -> None:
        """Initializes linear layer parameters using Xavier uniform distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, metrics: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
        """Generates adaptive cost weights for the current context.

        Args:
            metrics: current performance or environment metrics [batch_size, input_dim].
            day: current time indices [batch_size] (e.g., day of year).

        Returns:
            torch.Tensor: positive importance weights for reward components [batch_size, output_dim].
        """
        # Get time embeddings
        day_embed = self.time_embedding(day % self.n_days)

        # Concatenate metrics with time embeddings
        combined = torch.cat([metrics, day_embed], dim=1)

        # Generate raw weights
        raw_weights = self.layers(combined)

        # Apply activation to ensure positive weights
        weights = self.activation(raw_weights)

        return weights
