"""GNN Layer for NARGNN."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import BatchNorm
except ImportError:
    BatchNorm = None  # type: ignore


class GNNLayer(nn.Module):
    """Simplified GNN layer for NARGNN."""

    def __init__(self, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        """Initialize Class.

        Args:
            embed_dim (int): Description of embed_dim.
            act_fn (str): Description of act_fn.
            agg_fn (str): Description of agg_fn.
        """
        super().__init__()
        assert BatchNorm is not None, "torch_geometric required"

        self.embed_dim = embed_dim
        self.act_fn = getattr(nn.functional, act_fn)

        self.v_lin1 = nn.Linear(embed_dim, embed_dim)
        self.v_lin2 = nn.Linear(embed_dim, embed_dim)
        self.v_lin3 = nn.Linear(embed_dim, embed_dim)
        self.v_lin4 = nn.Linear(embed_dim, embed_dim)
        self.v_bn = BatchNorm(embed_dim)

        self.e_lin = nn.Linear(embed_dim, embed_dim)
        self.e_bn = BatchNorm(embed_dim)

        self.agg_fn = agg_fn

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x0, w0 = x, edge_attr

        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)

        edge_weighted = torch.sigmoid(w0) * x2[edge_index[1]]
        if self.agg_fn == "mean":
            from torch_scatter import scatter_mean

            aggregated = scatter_mean(edge_weighted, edge_index[0], dim=0, dim_size=x0.size(0))
        else:
            raise NotImplementedError(f"Aggregation {self.agg_fn} not implemented")

        x = x0 + self.act_fn(self.v_bn(x1 + aggregated))

        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return x, w
