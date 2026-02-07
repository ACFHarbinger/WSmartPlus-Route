"""
Problem-specific initialization embeddings.

Each problem type requires different input features to be embedded
before being processed by the encoder.
"""

from __future__ import annotations

import torch.nn as nn

from logic.src.models.embeddings.cvrpp import CVRPPInitEmbedding
from logic.src.models.embeddings.vrpp import VRPPInitEmbedding
from logic.src.models.embeddings.wcvrp import WCVRPInitEmbedding

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
            f"Unknown environment for embedding: {env_name}. Available: {list(INIT_EMBEDDING_REGISTRY.keys())}"
        )
    return INIT_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim)


__all__ = [
    "VRPPInitEmbedding",
    "CVRPPInitEmbedding",
    "WCVRPInitEmbedding",
    "INIT_EMBEDDING_REGISTRY",
    "get_init_embedding",
]
