"""
Heterogeneous Graph Neural Network Module.
"""

from typing import Dict, Tuple

import torch
from torch import nn

from logic.src.models.subnets.modules.normalization import Normalization

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, Linear, SAGEConv
except ImportError:
    HeteroConv = SAGEConv = Linear = object  # type: ignore
    HeteroData = None  # type: ignore


class HetGNNLayer(nn.Module):
    """
    Heterogeneous GNN Layer using HeteroConv.
    Capable of handling bipartite graphs or any multi-type graph.
    """

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int = 64,
        aggr: str = "sum",
        norm: str = "layer",
    ):
        """
        Initialize HetGNNLayer.

        Args:
            node_types: List of node types.
            edge_types: List of edge triples.
            hidden_dim: Hidden dimension.
            aggr: Aggregation scheme.
            norm: Normalization type.
        """
        super().__init__()

        if HeteroConv is object:
            raise ImportError("torch_geometric is required for HetGNNLayer")

        self.node_types = node_types
        self.edge_types = edge_types

        # Define convolution dictionary for each edge type
        # We use SAGEConv for simplicity as a general encoder
        conv_dict = {}
        for src, rel, dst in edge_types:
            # (in_channels_src, in_channels_dst) -> out_channels
            # We assume input features are already projected to hidden_dim or we use -1 for lazy initialization
            conv_dict[(src, rel, dst)] = SAGEConv((-1, -1), hidden_dim, aggr=aggr)

        self.conv = HeteroConv(conv_dict, aggr=aggr)

        # Normalization and activation per node type
        self.norms = nn.ModuleDict({nt: Normalization(hidden_dim, norm) for nt in node_types})
        self.acts = nn.ModuleDict({nt: nn.ReLU() for nt in node_types})

    def forward(
        self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Dictionary of node features {node_type: properties_tensor}
            edge_index_dict: Dictionary of edge indices {(src, rel, dst): edge_index_tensor}
        """
        # HeteroConv forward
        out_dict = self.conv(x_dict, edge_index_dict)

        # Apply Norm + Act + Residual (if shapes match)
        new_x_dict = {}
        for nt, x in out_dict.items():
            x = self.norms[nt](x)
            x = self.acts[nt](x)

            # Residual connection if dimensions match and it existed in input
            if nt in x_dict and x_dict[nt].shape == x.shape:
                new_x_dict[nt] = x + x_dict[nt]
            else:
                new_x_dict[nt] = x

        return new_x_dict
