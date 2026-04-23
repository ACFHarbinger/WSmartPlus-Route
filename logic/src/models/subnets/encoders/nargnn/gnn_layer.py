"""GNN Layer for NARGNN.

Attributes:
    GNNLayer: Simplified GNN layer for NARGNN.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn.gnn_layer import GNNLayer
    >>> layer = GNNLayer(embed_dim=128)
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import nn

try:
    from torch_scatter import scatter_mean
except (ImportError, OSError):
    scatter_mean = None

try:
    from torch_geometric.nn import BatchNorm
except (ImportError, OSError):
    BatchNorm = None


class GNNLayer(nn.Module):
    """Simplified GNN layer for NARGNN.

    This layer implements an anisotropic graph convolution where messages are
    gated by edge features.

    Attributes:
        embed_dim (int): Dimension of node and edge embeddings.
        act_fn (Callable): Activation function.
        v_lin1 (nn.Linear): Linear layer for node residual update.
        v_lin2 (nn.Linear): Linear layer for message generation.
        v_lin3 (nn.Linear): Linear layer for edge update (source node).
        v_lin4 (nn.Linear): Linear layer for edge update (target node).
        v_bn (nn.Module): Batch normalization for nodes.
        e_lin (nn.Linear): Linear layer for edge residual update.
        e_bn (nn.Module): Batch normalization for edges.
        agg_fn (str): Aggregation function name.
    """

    def __init__(self, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean") -> None:
        """Initializes the GNNLayer.

        Args:
            embed_dim: Dimension of node and edge embeddings.
            act_fn: Name of the activation function in torch.nn.functional.
            agg_fn: Aggregation function for message passing ("mean" supported).
        """
        super().__init__()
        self._check_dependencies()

        self.embed_dim = embed_dim
        self.act_fn: Callable = getattr(torch.nn.functional, act_fn)

        self.v_lin1 = nn.Linear(embed_dim, embed_dim)
        self.v_lin2 = nn.Linear(embed_dim, embed_dim)
        self.v_lin3 = nn.Linear(embed_dim, embed_dim)
        self.v_lin4 = nn.Linear(embed_dim, embed_dim)
        self.v_bn = BatchNorm(embed_dim)  # type: ignore[calling-non-callable]

        self.e_lin = nn.Linear(embed_dim, embed_dim)
        self.e_bn = BatchNorm(embed_dim)  # type: ignore[calling-non-callable]

        self.agg_fn = agg_fn

    def _check_dependencies(self) -> None:
        """Checks if required PyG dependencies are available.

        Raises:
            ImportError: If torch-geometric or torch-scatter is not available.
        """
        if BatchNorm is None or scatter_mean is None:
            raise ImportError(
                "GNNLayer requires torch-geometric and torch-scatter with matching CUDA extensions. "
                "Please ensure PyG is installed and working: "
                "https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GNN layer.

        Args:
            x: Node embeddings of shape (num_nodes, embed_dim).
            edge_index: Graph connectivity in COO format of shape (2, num_edges).
            edge_attr: Edge features of shape (num_edges, embed_dim).

        Returns:
            Tuple containing:
                - x (torch.Tensor): Updated node embeddings.
                - w (torch.Tensor): Updated edge features.

        Raises:
            NotImplementedError: If an unsupported aggregation function is used.
        """
        self._check_dependencies()
        x0, w0 = x, edge_attr

        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)

        edge_weighted = torch.sigmoid(w0) * x2[edge_index[1]]
        if self.agg_fn == "mean":
            aggregated = scatter_mean(edge_weighted, edge_index[0], dim=0, dim_size=x0.size(0))  # type: ignore[calling-non-callable]
        else:
            raise NotImplementedError(f"Aggregation {self.agg_fn} not implemented")

        x = x0 + self.act_fn(self.v_bn(x1 + aggregated))

        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return x, w
