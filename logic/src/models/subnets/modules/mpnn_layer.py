"""Message Passing Layer for MPNN.

This module provides the MessagePassingLayer, a core interaction layer for
Graph Neural Networks based on the Message Passing Neural Network (MPNN)
framework. It handles simultaneous node and edge attribute updates.

Attributes:
    MessagePassingLayer: Core interaction layer for Graph Neural Networks.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.mpnn_layer import MessagePassingLayer
    >>> layer = MessagePassingLayer(node_dim=128, edge_dim=64)
    >>> x = torch.randn(10, 128)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    >>> edge_attr = torch.randn(2, 64)
    >>> out_nodes, out_edges = layer(x, edge_index, edge_attr)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.modules.normalization import Normalization

if TYPE_CHECKING:
    from torch_geometric.nn import MessagePassing
else:
    try:
        from torch_geometric.nn import MessagePassing
    except (ImportError, OSError):
        MessagePassing = object


class MessagePassingLayer(MessagePassing):
    """Deep Message Passing Layer with coupled node and edge updates.

    Defines the structured interaction logic between node features and edge
    attributes, following a two-step update cycle:
        1. Edge Update: phi_e(h_i, h_j, e_ij) -> e'_ij
        2. Node Update: h_i + phi_v(h_i, sum(e'_ij)) -> h'_i

    Attributes:
        edge_model (nn.Sequential): MLP for transforming edge and neighbor features.
        node_model (nn.Sequential): MLP for final node state update from aggregates.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        """Initializes MessagePassingLayer.

        Args:
            node_dim: Dimensionality of input node features.
            edge_dim: Dimensionality of input/output edge features.
            hidden_dim: Hidden layer width for update MLPs.
            aggr: Neighborhood aggregation scheme ('add', 'mean', 'max').
            norm: Normalization strategy ('batch', 'layer', 'instance').
            bias: Whether to use bias parameters in internal linear layers.

        Raises:
            ImportError: If `torch_geometric` is not available in the environment.
        """
        if MessagePassing is object:
            raise ImportError("MessagePassingLayer requires torch_geometric to be installed.")

        super().__init__(aggr=aggr)

        # Edge update model (phi_e): (h_i, h_j, e_ij) -> e'_ij
        self.edge_model = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, norm),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim, bias=bias),
            Normalization(edge_dim, norm),
        )

        # Node update model (phi_v): (h_i, m_i) -> h'_i
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, norm),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim, bias=bias),
            Normalization(node_dim, norm),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Union[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Main interaction pass orchestrating updates.

        Args:
            args: Positional arguments (node features, indices, edge features).
            kwargs: Keyword arguments containing 'x', 'edge_index', and 'edge_attr'.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Any]: A tuple containing:
                - out_nodes: Updated node feature tensor.
                - edge_attr_updated: Updated edge attribute tensor.

        Raises:
            ValueError: If input tensors are missing from arguments.
        """
        # Extract specific tensors from kwargs or args
        x = kwargs.get("x", args[0] if len(args) > 0 else None)
        edge_index = kwargs.get("edge_index", args[1] if len(args) > 1 else None)
        edge_attr = kwargs.get("edge_attr", args[2] if len(args) > 2 else None)

        if x is None or edge_index is None or edge_attr is None:
            raise ValueError("Forward requires x, edge_index, and edge_attr")

        row, col = edge_index

        # 1. Update edges
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_updated = self.edge_model(edge_input)

        # 2. Update nodes
        out_nodes = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)

        return out_nodes, edge_attr_updated

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Transforms neighboring node features into messages.

        Args:
            x_i: Features of target nodes (receivers).
            x_j: Features of source nodes (senders).
            edge_attr: Edge attributes on the incident edges.

        Returns:
            torch.Tensor: The formulated message vector.
        """
        # Note: x_i, x_j are provided by PyG propagate mechanism
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Updates node features using neighborhood message aggregates.

        Args:
            aggr_out: Neighborhood message summation/aggregation result.
            x: Original node features from before propagation.

        Returns:
            torch.Tensor: Final updated node feature representation.
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_model(node_input)
