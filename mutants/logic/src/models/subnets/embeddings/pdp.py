"""
PDP Embedding.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict


class PDPInitEmbedding(nn.Module):
    """
    Initial embedding for Pickup and Delivery Problem (PDP).

    Projects Depot, Pickup nodes, and Delivery nodes into the embedding space
    using separate linear projections to account for heterogeneity.
    """

    def __init__(self, embed_dim: int, linear_bias: bool = True):
        """
        Initialize PDPInitEmbedding.

        Args:
            embed_dim: Embedding dimension.
            linear_bias: Whether to use bias.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # separate projections for depot, pickup and delivery
        self.proj_depot = nn.Linear(2, embed_dim, bias=linear_bias)
        self.proj_pickup = nn.Linear(2, embed_dim, bias=linear_bias)
        self.proj_delivery = nn.Linear(2, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Args:
            td: TensorDict containing 'locs' (batch, 2*N, 2) and 'depot' (batch, 2)

        Returns:
            embeddings: (batch, 2*N+1, embed_dim) [Depot, Pickups, Deliveries]
        """
        locs = td["locs"]
        depot = td["depot"]

        # Combine depot and locs? No, logic is usually separate for efficient projection
        # locs contains [Pickup_1, ..., Pickup_N, Delivery_1, ..., Delivery_N]

        batch_size, num_total, _ = locs.shape
        num_pairs = num_total // 2

        pickups = locs[:, :num_pairs, :]
        deliveries = locs[:, num_pairs:, :]

        # Project
        h_depot = self.proj_depot(depot.unsqueeze(1))  # (batch, 1, embed_dim)
        h_pickup = self.proj_pickup(pickups)  # (batch, N, embed_dim)
        h_delivery = self.proj_delivery(deliveries)  # (batch, N, embed_dim)

        # Concatenate: [Depot, Pickups, Deliveries]
        out = torch.cat([h_depot, h_pickup, h_delivery], dim=1)

        return out
