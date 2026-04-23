"""TSP Edge embedding module.

This module provides the TSPEdgeEmbedding layer, which constructs Euclidean-based
graph representations for Traveling Salesman Problems and related variants.

Attributes:
    TSPEdgeEmbedding: Euclidean distance edge encoder for TSP-family problems.

Example:
    >>> from logic.src.models.subnets.embeddings.edges.tsp import TSPEdgeEmbedding
    >>> embedder = TSPEdgeEmbedding(embed_dim=128)
    >>> pyg_batch = embedder(td, init_embeddings)
"""

from __future__ import annotations

from .base import EdgeEmbedding


class TSPEdgeEmbedding(EdgeEmbedding):
    """Edge embedding for TSP-like problems.

    Uses pairwise Euclidean distances as the primary edge features, optionally
    sparsified to a k-nearest neighbor adjacency structure per node to reduce
    graph complexity.

    Attributes:
        node_dim (int): Dimensionality of raw distance features (fixed to 1).
    """

    node_dim: int = 1
