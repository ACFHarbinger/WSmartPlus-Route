"""MLP layer implementation.

Attributes:
    MLPLayer: Simple MLP layer with configurable activation and normalization.

Example:
    >>> from logic.src.models.subnets.encoders.mlp.mlp_layer import MLPLayer
    >>> from logic.src.configs.models.normalization import NormalizationConfig
    >>> layer = MLPLayer(hidden_dim=128, norm_config=NormalizationConfig())
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.models.subnets.modules.normalization import Normalization


class MLPLayer(nn.Module):
    """Simple MLP layer with configurable activation and normalization.

    This layer performs: Linear → Normalization → Activation → Residual Connection.

    Attributes:
        hidden_dim (int): Hidden dimension size for the linear transformation.
        U (nn.Linear): Linear transformation layer.
        norm (Normalization): Normalization layer.
        activation (ActivationFunction): Activation function module.
    """

    def __init__(
        self,
        hidden_dim: int,
        norm_config: NormalizationConfig,
        activation_config: Optional[ActivationConfig] = None,
    ) -> None:
        """Initializes the MLPLayer.

        Args:
            hidden_dim: Hidden dimension size.
            norm_config: Normalization configuration.
            activation_config: Activation function configuration. Defaults to ReLU if None.
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim

        # Linear transformation
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Normalization
        self.norm = Normalization(
            hidden_dim,
            norm_config.norm_type,
            norm_config.epsilon,
            norm_config.learn_affine,
            norm_config.track_stats,
            norm_config.momentum,
            norm_config.n_groups,
            norm_config.k_lrnorm,
        )

        # Activation function (default to ReLU if no config provided)
        if activation_config is None:
            activation_config = ActivationConfig(name="relu")

        self.activation = ActivationFunction(activation_config=activation_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP layer.

        Args:
            x: Input tensor of shape (batch_size, num_nodes, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, hidden_dim).
        """
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        x = self.norm(x.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Apply activation
        x = self.activation(x)

        # Residual connection
        x = x_in + x

        return x
