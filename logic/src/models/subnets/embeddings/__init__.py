"""
Problem-specific initialization embeddings.

Each problem type requires different input features to be embedded
before being processed by the encoder.
"""

from __future__ import annotations

import torch.nn as nn

from logic.src.models.subnets.embeddings.cvrpp import CVRPPInitEmbedding
from logic.src.models.subnets.embeddings.pdp import PDPInitEmbedding
from logic.src.models.subnets.embeddings.vrpp import VRPPInitEmbedding
from logic.src.models.subnets.embeddings.wcvrp import WCVRPInitEmbedding

from .context import (
    ContextEmbedder,
    CVRPPContext,
    EnvContext,
    GenericContextEmbedder,
    SWCVRPContext,
    VRPPContext,
    VRPPContextEmbedder,
    WCContextEmbedder,
    WCVRPContext,
)
from .dynamic import DynamicEmbedding
from .edges import (
    EDGE_EMBEDDING_REGISTRY,
    CVRPPEdgeEmbedding,
    EdgeEmbedding,
    NoEdgeEmbedding,
    TSPEdgeEmbedding,
    WCVRPEdgeEmbedding,
    get_edge_embedding,
)
from .static import StaticEmbedding

# Embedding registry
INIT_EMBEDDING_REGISTRY = {
    "vrpp": VRPPInitEmbedding,
    "cvrpp": CVRPPInitEmbedding,
    "wcvrp": WCVRPInitEmbedding,
    "cwcvrp": WCVRPInitEmbedding,
    "sdwcvrp": WCVRPInitEmbedding,
    "swcvrp": WCVRPInitEmbedding,
    "scwcvrp": WCVRPInitEmbedding,
    "pdp": PDPInitEmbedding,
}

CONTEXT_EMBEDDING_REGISTRY = {
    "vrpp": VRPPContext,
    "cvrpp": CVRPPContext,
    "wcvrp": WCVRPContext,
    "cwcvrp": WCVRPContext,
    "sdwcvrp": WCVRPContext,
    "swcvrp": SWCVRPContext,
    "scwcvrp": SWCVRPContext,
}

DYNAMIC_EMBEDDING_REGISTRY = {
    "static": StaticEmbedding,
    "dynamic": DynamicEmbedding,
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
    "EnvContext",
    "VRPPContext",
    "CVRPPContext",
    "WCVRPContext",
    "SWCVRPContext",
    "ContextEmbedder",
    "VRPPContextEmbedder",
    "WCContextEmbedder",
    "DynamicEmbedding",
    "StaticEmbedding",
    "EdgeEmbedding",
    "TSPEdgeEmbedding",
    "CVRPPEdgeEmbedding",
    "WCVRPEdgeEmbedding",
    "NoEdgeEmbedding",
    "INIT_EMBEDDING_REGISTRY",
    "CONTEXT_EMBEDDING_REGISTRY",
    "DYNAMIC_EMBEDDING_REGISTRY",
    "EDGE_EMBEDDING_REGISTRY",
    "get_init_embedding",
    "get_edge_embedding",
    "GenericContextEmbedder",
]
