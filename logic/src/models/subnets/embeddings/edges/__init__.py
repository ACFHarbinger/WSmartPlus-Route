"""
Edge embedding modules for graph-based encoders.
"""

from .base import EdgeEmbedding
from .cvrpp import CVRPPEdgeEmbedding
from .none import NoEdgeEmbedding
from .registry import EDGE_EMBEDDING_REGISTRY, get_edge_embedding
from .tsp import TSPEdgeEmbedding
from .wcvrp import WCVRPEdgeEmbedding

__all__ = [
    "EdgeEmbedding",
    "CVRPPEdgeEmbedding",
    "NoEdgeEmbedding",
    "TSPEdgeEmbedding",
    "WCVRPEdgeEmbedding",
    "EDGE_EMBEDDING_REGISTRY",
    "get_edge_embedding",
]
