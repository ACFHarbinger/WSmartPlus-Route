"""WCVRP Embedding module.

This module provides the WCVRPInitEmbedding layer, which encodes locations
and fill levels for Waste Collection Vehicle Routing Problems.

Attributes:
    WCVRPInitEmbedding: Initial feature encoder for WCVRP instances.

Example:
    >>> from logic.src.models.subnets.embeddings.wcvrp import WCVRPInitEmbedding
    >>> embed = WCVRPInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class WCVRPInitEmbedding(nn.Module):
    """Initial embedding for WCVRP problems.

    Projects static node features (coordinates and container fill levels)
    into a high-dimensional embedding space.

    Attributes:
        node_embed (nn.Linear): Linear projection for customer node features.
        depot_embed (nn.Linear): Specialized linear projection for depot location.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        """Initializes WCVRPInitEmbedding.

        Args:
            embed_dim: Internal embedding dimensionality.
        """
        super().__init__()
        # Node features: x, y, waste (fill_level)
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """Encodes WCVRP instance features into initial node embeddings.

        Args:
            td: TensorDict containing instance metadata ('locs', 'waste', 'depot').

        Returns:
            torch.Tensor: Initial node embeddings of shape (batch, num_nodes, dim).
        """
        locs = td["locs"]
        waste = td.get("waste")  # Fill levels

        if waste is None:
            waste = torch.zeros(td.batch_size[0], locs.size(1), device=td.device)

        node_features = torch.cat([locs, waste.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings
