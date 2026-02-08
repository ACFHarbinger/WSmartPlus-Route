"""
VRPP Embedding module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class VRPPInitEmbedding(nn.Module):
    """Initial embedding for VRPP problems."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize VRPPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        # Node features: x, y, prize
        self.node_embed = nn.Linear(3, embed_dim)
        # Depot features: x, y
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Embed VRPP instance.

        Args:
            td: TensorDict with keys: locs, depot, prize

        Returns:
            Node embeddings [batch, num_nodes, embed_dim]
        """
        locs = td["locs"]  # [batch, num_nodes, 2]
        prize = td.get("waste", td.get("prize", td.get("demand")))

        if prize is None:
            # Fallback if missing
            prize = torch.zeros(td.batch_size[0], locs.size(1), device=td.device)

        # Combine location and prize (waste) for non-depot nodes
        node_features = torch.cat([locs, prize.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        # Special embedding for depot
        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
