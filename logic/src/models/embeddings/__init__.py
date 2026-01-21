"""
Problem-specific initialization embeddings.

Each problem type requires different input features to be embedded
before being processed by the encoder.
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
            embed_dim: Description for embed_dim.
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
        prize = td["prize"]  # [batch, num_nodes]

        # Combine location and prize for non-depot nodes
        node_features = torch.cat([locs, prize.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        # Special embedding for depot
        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings


class CVRPPInitEmbedding(nn.Module):
    """Initial embedding for CVRPP (capacitated VRPP)."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize VRPPInitEmbedding.

        Args:
            embed_dim: Description for embed_dim.
        """
        super().__init__()
        # Node features: x, y, prize, demand
        self.node_embed = nn.Linear(4, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward.

        Args:
            td: Description for td.
            prize.unsqueeze(-1): Description for prize.unsqueeze(-1).
            demand.unsqueeze(-1)]: Description for demand.unsqueeze(-1)].
            dim: Description for dim.
            0]: Description for 0].

        Returns:
            Computation result.
        """
        locs = td["locs"]
        prize = td["prize"]
        demand = td["demand"]

        node_features = torch.cat([locs, prize.unsqueeze(-1), demand.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings


class WCVRPInitEmbedding(nn.Module):
    """Initial embedding for WCVRP problems."""

    def __init__(self, embed_dim: int = 128):
        """
        Initialize VRPPInitEmbedding.

        Args:
            embed_dim: Description for embed_dim.
        """
        super().__init__()
        # Node features: x, y, fill_level
        self.node_embed = nn.Linear(3, embed_dim)
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: TensorDict) -> torch.Tensor:
        """
        Forward.

        Args:
            None.

        Returns:
            Computation result.
        """
        locs = td["locs"]
        demand = td["demand"]  # Fill levels

        node_features = torch.cat([locs, demand.unsqueeze(-1)], dim=-1)
        embeddings = self.node_embed(node_features)

        depot_emb = self.depot_embed(td["depot"])
        embeddings[:, 0] = depot_emb

        return embeddings


# Embedding registry
INIT_EMBEDDING_REGISTRY = {
    "vrpp": VRPPInitEmbedding,
    "cvrpp": CVRPPInitEmbedding,
    "wcvrp": WCVRPInitEmbedding,
    "cwcvrp": WCVRPInitEmbedding,
    "sdwcvrp": WCVRPInitEmbedding,
}


def get_init_embedding(env_name: str, embed_dim: int = 128) -> nn.Module:
    """
    Get problem-specific initial embedding layer.

    Args:
        env_name: Environment/problem name.
        embed_dim: Embedding dimension.

    Returns:
        Initialized embedding module.
    """
    if env_name not in INIT_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown environment for embedding: {env_name}. " f"Available: {list(INIT_EMBEDDING_REGISTRY.keys())}"
        )
    return INIT_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim)


__all__ = [
    "VRPPInitEmbedding",
    "CVRPPInitEmbedding",
    "WCVRPInitEmbedding",
    "INIT_EMBEDDING_REGISTRY",
    "get_init_embedding",
]
