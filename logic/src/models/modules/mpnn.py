from typing import Tuple

import torch
import torch.nn as nn

from logic.src.models.modules.normalization import Normalization

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import MessagePassing
except ImportError:
    MessagePassing = object
    Batch = Data = None


class MessagePassingLayer(MessagePassing):
    """
    Message Passing Layer with custom node and edge update models.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension for MLPs
        aggr: Aggregation scheme ('add', 'mean', 'max')
        norm: Normalization type ('batch', 'layer', etc.)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
        bias: bool = True,
    ):
        if MessagePassing is object:
            raise ImportError("torch_geometric is required for MessagePassingLayer")

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
        # m_i is aggregated message
        self.node_model = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, norm),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim, bias=bias),
            Normalization(node_dim, norm),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # propagate performs message passing:
        # 1. message() creates messages
        # 2. aggregate() aggregates messages
        # 3. update() updates node embeddings

        # We also want to update edge attributes, so we calculate them first manually if needed,
        # but PyG `propagate` flow is centered on nodes.
        # Standard MPNN often updates edges then nodes.

        row, col = edge_index

        # 1. Update edges
        # Start node features, End node features, Edge features
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_updated = self.edge_model(edge_input)

        # 2. Update nodes
        # We pass the *updated* edge attributes to propagate
        out_nodes = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)

        return out_nodes, edge_attr_updated

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Message function: In this MPNN variant, the message is just the updated edge feature
        # or processed edge feature.
        # Here we just pass the updated edge attribute as the message to be aggregated.
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Node update step: Combine aggregated messages with current node features
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_model(node_input)


class MPNNEncoder(nn.Module):
    """
    Encoder stack using MessagePassingLayer.
    """

    def __init__(
        self,
        num_layers: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        aggr: str = "add",
        norm: str = "batch",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_dim=node_dim if i == 0 else hidden_dim,  # Assuming dims stay constant usually
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    aggr=aggr,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )

        # Project dimensions if needed, here assuming constant dim for simplicity of the stack
        # If node_dim != hidden_dim, we might need a projection first.
        self.input_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else nn.Identity()

        # Re-create layers with consistent hidden_dim
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_dim=hidden_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim, aggr=aggr, norm=norm
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for layer in self.layers:
            # Residual connection could be added here
            x_new, edge_attr_new = layer(x, edge_index, edge_attr)
            x = x + x_new  # Residual
            edge_attr = edge_attr + edge_attr_new  # Residual

        return x, edge_attr
