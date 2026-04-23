"""WCVRP Edge embedding module.

This module provides the WCVRPEdgeEmbedding layer, which constructs Euclidean-based
graph representations specifically for Waste Collection VRP variants.

Attributes:
    WCVRPEdgeEmbedding: Edge encoder for waste collection routing problems.

Example:
    >>> from logic.src.models.subnets.embeddings.edges.wcvrp import WCVRPEdgeEmbedding
    >>> embedder = WCVRPEdgeEmbedding(embed_dim=128)
    >>> pyg_batch = embedder(td, init_embeddings)
"""

from __future__ import annotations

from .cvrpp import CVRPPEdgeEmbedding


class WCVRPEdgeEmbedding(CVRPPEdgeEmbedding):
    """Edge embedding for Waste Collection VRP problems.

    Inherits the structural logic from CVRPPEdgeEmbedding, ensuring critical
    depot-to-customer connectivity is maintained during sparsification.

    Attributes:
        node_dim (int): Dimensionality of raw distance features.
    """

    pass
