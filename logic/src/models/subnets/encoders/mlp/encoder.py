"""MLP Encoder.

Attributes:
    MLPEncoder: Simple MLP encoder with configurable activation and normalization.

Example:
    >>> from logic.src.models.subnets.encoders.mlp import MLPEncoder
    >>> encoder = MLPEncoder(n_layers=3, feed_forward_hidden=128)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig

from .mlp_layer import MLPLayer


class MLPEncoder(nn.Module):
    """Simple MLP encoder with configurable activation and normalization.

    This encoder consists of stacked MLPLayer instances, each performing:
    Linear → Normalization → Activation → Residual Connection.

    Unlike graph-based encoders, this encoder operates independently of graph
    structure and can be used for simple feature transformations.

    Attributes:
        norm_config (NormalizationConfig): Normalization configuration.
        activation_config (ActivationConfig): Activation function configuration.
        n_layers (int): Number of layers in the encoder.
        hidden_dim (int): Hidden dimension size for each layer.
        layers (nn.ModuleList): Stack of MLP layers.
    """

    def __init__(
        self,
        n_layers: int,
        feed_forward_hidden: int,
        norm: str = "layer",
        learn_affine: bool = True,
        track_norm: bool = False,
        activation: str = "relu",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the MLPEncoder.

        Args:
            n_layers: Number of MLP layers to stack.
            feed_forward_hidden: Hidden dimension size for each layer.
            norm: Normalization type ("batch", "layer", "instance", "group", or None).
            learn_affine: Whether to learn affine parameters in normalization layers.
            track_norm: Whether to track running statistics in normalization.
            activation: Activation function name.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
        """
        super(MLPEncoder, self).__init__()

        # Create normalization config
        self.norm_config = NormalizationConfig(
            norm_type=norm,
            learn_affine=learn_affine,
            track_stats=track_norm,
        )

        # Create activation config
        self.activation_config = ActivationConfig(name=activation)

        # Store configuration
        self.n_layers = n_layers
        self.hidden_dim = feed_forward_hidden

        # Build layer stack
        self.layers = nn.ModuleList(
            [
                MLPLayer(
                    feed_forward_hidden,
                    self.norm_config,
                    self.activation_config,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, graph: Optional[Any] = None) -> torch.Tensor:
        """Forward pass through all MLP layers.

        Args:
            x: Input node features of shape (batch_size, num_nodes, hidden_dim).
            graph: Optional graph structure (unused by MLP encoder, included for API compatibility).

        Returns:
            torch.Tensor: Transformed node features of shape (batch_size, num_nodes, hidden_dim).
        """
        # Note: graph parameter is unused but kept for API compatibility with other encoders
        for layer in self.layers:
            x = layer(x)

        return x
