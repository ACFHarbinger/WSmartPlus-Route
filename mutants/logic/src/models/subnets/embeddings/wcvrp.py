"""
WCVRP Embedding module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class WCVRPInitEmbedding(nn.Module):
    """Initial embedding for WCVRP problems."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize WCVRPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, fill_level
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward pass for WCVRP embedding.

        Args:
            td: TensorDict containing 'locs', 'demand' (fill levels), 'depot'.

        Returns:
            Embeddings tensor [batch, num_nodes, embed_dim].
        """
        locs = td["locs"]
        demand = td.get("waste", td.get("demand"))  # Fill levels

        if demand is None:
            demand = torch.zeros(td.batch_size[0], locs.size(1), device=td.device)

        node_features = torch.cat([locs, demand.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
