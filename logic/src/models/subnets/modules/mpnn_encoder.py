"""MPNN Encoder stack.

This module provides the MPNNEncoder, which implements a deep graph encoder
composed of multiple message-passing layers. It projects initial node and edge
features into a unified hidden space before iteratively updating them.

Attributes:
    MPNNEncoder: Stack of message passing layers for graph encoding.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.mpnn_encoder import MPNNEncoder
    >>> encoder = MPNNEncoder(num_layers=3, node_dim=12, edge_dim=2)
    >>> x = torch.randn(10, 12)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    >>> edge_attr = torch.randn(2, 2)
    >>> x_enc, e_enc = encoder(x, edge_index, edge_attr)
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .mpnn_layer import MessagePassingLayer


class MPNNEncoder(nn.Module):
    """Deep graph encoder using sequentially stacked MessagePassingLayers.

    This module handles the initial dimensionality projection of node and edge attributes
    and manages the iterative information flow through multiple graph convolution steps,
    incorporating residual connections at each stage.

    Attributes:
        input_proj (nn.Module): Linear layer to unify node feature dimensions.
        edge_proj (nn.Module): Linear layer to unify edge feature dimensions.
        layers (nn.ModuleList): Sequential collection of message passing updates.
    """

    def __init__(
        self,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
    ) -> None:
        """Initializes MPNNEncoder.

        Args:
            num_layers: Total number of message passing blocks to stack.
            node_dim: Input dimensionality of raw node features.
            edge_dim: Input dimensionality of raw edge features.
            hidden_dim: Target dimensionality for all hidden representations.
            aggr: Neighborhood aggregation scheme ('add', 'mean', 'max').
            norm: Normalization strategy to apply within layers ('batch', 'layer').
        """
        super().__init__()

        # Project dimensions if needed
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    aggr=aggr,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes graph features through iterative message passing.

        Args:
            x: Node feature tensor of shape (num_nodes, node_dim).
            edge_index: Graph adjacency spectral matching (2, num_edges).
            edge_attr: Edge feature tensor of shape (num_edges, edge_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Encoded node features of shape (num_nodes, hidden_dim).
                - Encoded edge features of shape (num_edges, hidden_dim).
        """
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual
            edge_attr = edge_attr + edge_attr_new  # Residual

        return x, edge_attr
