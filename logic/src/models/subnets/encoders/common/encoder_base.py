"""Base encoder classes for reducing boilerplate across encoder implementations.

This module provides abstract base classes that handle common encoder patterns:
- Parameter initialization (n_heads, embed_dim, n_layers, normalization, activation)
- Layer stacking via ModuleList
- Standard forward pass with dropout
- Optional hyper-connection support

Subclasses only need to implement `_create_layer()` to define their specific layer type.

Attributes:
    TransformerEncoderBase: Abstract base class for transformer-style graph encoders.

Example:
    >>> from logic.src.models.subnets.encoders.common.encoder_base import TransformerEncoderBase
    >>> # Subclasses would implement _create_layer and then be instantiated
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig


class TransformerEncoderBase(nn.Module, ABC):
    """Abstract base class for transformer-style graph encoders.

    Handles common boilerplate such as parameter validation, layer stacking,
    standard forward pass patterns, and optional hyper-connection handling.

    Attributes:
        n_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        n_layers (int): Number of encoder layers.
        feed_forward_hidden (int): Hidden dimension for feed-forward layers.
        dropout_rate (float): Dropout probability.
        conn_type (str): Connection type (e.g., "skip", "dense", "hyper").
        expansion_rate (int): Expansion factor for hyper-connections.
        norm_config (NormalizationConfig): Normalization configuration.
        activation_config (ActivationConfig): Activation function configuration.
        kwargs (Dict[str, Any]): Additional keyword arguments for layer creation.
        layers (nn.ModuleList): Stack of encoder layers.
        dropout (nn.Dropout): Dropout layer applied after encoding.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initializes the TransformerEncoderBase.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension (must be divisible by n_heads).
            n_layers: Number of encoder layers to stack.
            feed_forward_hidden: Hidden dimension for feed-forward sublayers.
            norm_config: Normalization configuration.
            activation_config: Activation function configuration.
            dropout_rate: Dropout probability applied after layers.
            connection_type: Connection type ("skip", "dense", or "hyper").
            expansion_rate: Expansion factor for hyper-connections.
            kwargs: Additional keyword arguments for layer creation.

        Raises:
            ValueError: If embed_dim is not divisible by n_heads.
        """
        super(TransformerEncoderBase, self).__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.feed_forward_hidden = feed_forward_hidden
        self.dropout_rate = dropout_rate
        self.conn_type = connection_type
        self.expansion_rate = expansion_rate
        self.kwargs = kwargs

        self.norm_config = norm_config if norm_config is not None else NormalizationConfig()
        self.activation_config = activation_config if activation_config is not None else ActivationConfig()

        if n_heads > 0 and embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}). "
                f"Got remainder: {embed_dim % n_heads}"
            )

        self.layers = nn.ModuleList([self._create_layer(layer_idx) for layer_idx in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    @abstractmethod
    def _create_layer(self, layer_idx: int) -> nn.Module:
        """Creates a single encoder layer.

        Args:
            layer_idx: Index of the layer being created (0 to n_layers-1).

        Returns:
            nn.Module: The created encoder layer.
        """
        raise NotImplementedError("Subclasses must implement _create_layer()")

    def forward(
        self,
        x: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard encoder forward pass with layer stacking.

        Args:
            x: Input node features of shape (batch_size, num_nodes, embed_dim).
            edges: Edge information (adjacency, features, etc.).
            mask: Attention mask of shape (batch, nodes, nodes).

        Returns:
            torch.Tensor: Encoded node features of shape (batch_size, num_nodes, embed_dim).
        """
        if "hyper" in self.conn_type:
            H = x.unsqueeze(-1).repeat(1, 1, 1, self.expansion_rate)
            curr = H
        else:
            curr = x

        for layer in self.layers:
            if edges is not None:
                curr = layer(curr, edges=edges)
            elif mask is not None:
                curr = layer(curr, mask=mask)
            else:
                curr = layer(curr)

        if "hyper" in self.conn_type:
            curr = curr.mean(dim=-1)

        output: torch.Tensor = self.dropout(curr)
        return output
