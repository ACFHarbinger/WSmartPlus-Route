"""
CVRPP Embedding module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class CVRPPInitEmbedding(nn.Module):
    """Initial embedding for CVRPP (capacitated VRPP)."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize VRPPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, prize, demand
        self.node_embed = nn.Linear(4, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward pass for CVRPP embedding.

        Args:
            td: TensorDict containing 'locs', 'prize', 'demand', 'depot'.

        Returns:
            Embeddings tensor [batch, num_nodes, embed_dim].
        """
        locs = td["locs"]
        prize = td.get("waste", td.get("prize", torch.zeros(td.batch_size[0], locs.size(1), device=td.device)))
        demand = td.get("demand", torch.zeros_like(prize))

        node_features = torch.cat([locs, prize.unsqueeze(-1), demand.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
