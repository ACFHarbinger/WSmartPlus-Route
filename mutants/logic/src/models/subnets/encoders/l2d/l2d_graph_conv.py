"""Graph Convolution Layer for L2D."""

from __future__ import annotations

import torch
import torch.nn as nn


class L2DGraphConv(nn.Module):
    """
    Graph Convolution Layer for L2D.

    Processes two types of edges:
    1. Precedence/Conjunctive (Job sequence)
    2. Machine/Disjunctive (Same machine)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128, aggregation: str = "mean"):
        """
        Initialize L2DGraphConv.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden dimension.
            aggregation: Aggregation type.
        """
        super().__init__()
        self.aggregation = aggregation

        # Message passing layers
        self.proj_prec = nn.Linear(embed_dim, embed_dim)
        self.proj_mach = nn.Linear(embed_dim, embed_dim)

        # Update layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # concat(h, agg_prec, agg_mach)
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor, adj_prec: torch.Tensor, adj_mach: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Node embeddings (batch, num_nodes, embed_dim)
            adj_prec: Precedence adjacency (batch, num_nodes, num_nodes) - normalized row-wise
            adj_mach: Machine adjacency (batch, num_nodes, num_nodes) - normalized row-wise
        """
        # Message passing
        # 1. Precedence neighbors
        msg_prec = self.proj_prec(h)
        agg_prec = torch.bmm(adj_prec, msg_prec)

        # 2. Machine neighbors
        msg_mach = self.proj_mach(h)
        agg_mach = torch.bmm(adj_mach, msg_mach)

        # 3. Update
        combined = torch.cat([h, agg_prec, agg_mach], dim=-1)
        out = self.mlp(combined)

        return self.norm(h + out)  # Residual connection
