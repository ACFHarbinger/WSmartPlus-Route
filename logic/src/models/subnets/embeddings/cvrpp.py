"""CVRPP Embedding module.

This module provides the CVRPPInitEmbedding layer, which encodes locations
and waste levels for Capacitated Vehicle Routing Problems with Profits.

Attributes:
    CVRPPInitEmbedding: Initial feature encoder for CVRPP instances.

Example:
    >>> from logic.src.models.subnets.embeddings.cvrpp import CVRPPInitEmbedding
    >>> embed = CVRPPInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class CVRPPInitEmbedding(nn.Module):
    """Initial embedding for CVRPP (Capacitated VRPP).

    Projects static node features (coordinates and waste/demand quantities)
    into a joint embedding space.

    Attributes:
        node_embed (nn.Linear): Linear projection for customer node features.
        depot_embed (nn.Linear): Specialized linear projection for depot location.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        """Initializes CVRPPInitEmbedding.

        Args:
            embed_dim: Internal embedding dimensionality.
        """
        super().__init__()
        # Node features: x, y, waste
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """Encodes CVRPP features into initial node embeddings.

        Args:
            td: TensorDict containing instance metadata ('locs', 'waste', 'depot').

        Returns:
            torch.Tensor: Initial node embeddings of shape (batch, num_nodes, dim).
        """
        locs = td["locs"]
        waste = td.get("waste", torch.zeros(td.batch_size[0], locs.size(1), device=td.device))

        node_features = torch.cat([locs, waste.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
