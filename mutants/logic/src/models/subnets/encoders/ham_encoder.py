"""
HAM Encoder.
"""

from typing import Tuple

import torch
import torch.nn as nn
from logic.src.models.modules.ham_attention import HeterogeneousAttentionLayer


class HAMEncoder(nn.Module):
    """
    Heterogeneous Attention Model (HAM) Encoder.

    Encodes a PDP graph by handling Depot, Pickup, and Delivery nodes as distinct types
    and engaging in cross-attention message passing between them.
    Expects heterogeneous embeddings to be concatenated: [Depot, Pickups, Deliveries].
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
    ):
        """
        Initialize HAMEncoder.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of layers.
            feedforward_hidden: Hidden dimension.
            normalization: Normalization type.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Node types and Edge types for the Heterogeneous Attention
        self.node_types = ["depot", "pickup", "delivery"]
        # Fully connected heterogeneity: Everyone attends to everyone
        self.edge_types = [(src, "to", dst) for src in self.node_types for dst in self.node_types]

        self.layers = nn.ModuleList(
            [
                HeterogeneousAttentionLayer(
                    node_types=self.node_types,
                    edge_types=self.edge_types,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, embeddings: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (batch, 2*N+1, embed_dim) - [Depot, Pickups, Deliveries]

        Returns:
            embeddings: (batch, 2*N+1, embed_dim) - Encoded features
            init_embedding: (batch, 2*N+1, embed_dim) - Initial embeddings
        """

        batch_size, num_total_nodes, _ = embeddings.shape
        num_pairs = (num_total_nodes - 1) // 2

        # Split: Depot (0), Pickups (1..N), Deliveries (N+1..2N)
        h_depot = embeddings[:, 0:1, :]
        h_pickup = embeddings[:, 1 : num_pairs + 1, :]
        h_delivery = embeddings[:, num_pairs + 1 :, :]

        x_dict = {"depot": h_depot, "pickup": h_pickup, "delivery": h_delivery}

        # Save initial embedding for return
        init_h = embeddings

        # Heterogeneous Attention Layers
        for layer in self.layers:
            x_dict = layer(x_dict)

        # Recombine
        out_depot = x_dict["depot"]
        out_pickup = x_dict["pickup"]
        out_delivery = x_dict["delivery"]

        out_embeddings = torch.cat([out_depot, out_pickup, out_delivery], dim=1)

        return out_embeddings, init_h
