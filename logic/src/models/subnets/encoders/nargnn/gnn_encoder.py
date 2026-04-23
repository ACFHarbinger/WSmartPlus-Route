"""Simplified GNN Encoder for NARGNN.

Attributes:
    SimplifiedGNNEncoder: Simplified GNN encoder for NARGNN.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn.gnn_encoder import SimplifiedGNNEncoder
    >>> enc = SimplifiedGNNEncoder(num_layers=3, embed_dim=128)
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import nn

from .gnn_layer import GNNLayer


class SimplifiedGNNEncoder(nn.Module):
    """Simplified GNN encoder for NARGNN.

    This encoder stack consists of multiple GNN layers that process node and
    edge features through message passing.

    Attributes:
        act_fn (Callable): Activation function.
        layers (nn.ModuleList): Stack of GNN layers.
    """

    def __init__(self, num_layers: int, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean") -> None:
        """Initializes the SimplifiedGNNEncoder.

        Args:
            num_layers: Number of GNN layers in the encoder stack.
            embed_dim: Dimension of node and edge embeddings.
            act_fn: Name of the activation function in torch.nn.functional.
            agg_fn: Aggregation function for message passing.
        """
        super().__init__()
        self.act_fn: Callable = getattr(torch.nn.functional, act_fn)
        self.layers = nn.ModuleList([GNNLayer(embed_dim, act_fn, agg_fn) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GNN encoder stack.

        Args:
            x: Node embeddings of shape (num_nodes, embed_dim).
            edge_index: Graph connectivity in COO format.
            edge_attr: Edge features of shape (num_edges, embed_dim).

        Returns:
            Tuple containing:
                - x (torch.Tensor): Final node embeddings.
                - edge_attr (torch.Tensor): Final edge features.
        """
        x = self.act_fn(x)
        edge_attr = self.act_fn(edge_attr)
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
