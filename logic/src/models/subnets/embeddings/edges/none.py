"""Null edge embedding module.

This module provides the NoEdgeEmbedding layer, which constructs fully-connected
graphs with zero-valued edge attributes for problems where edge features are ignored.

Attributes:
    NoEdgeEmbedding: Placeholder edge encoder for non-graph tasks.

Example:
    >>> from logic.src.models.subnets.embeddings.edges.none import NoEdgeEmbedding
    >>> embedder = NoEdgeEmbedding(embed_dim=128)
    >>> pyg_batch = embedder(td, init_embeddings)
"""

from __future__ import annotations

from typing import Any, List

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

from logic.src.utils.ops import get_full_graph_edge_index


class NoEdgeEmbedding(nn.Module):
    """Dummy edge embedding for problems that don't use edge features.

    Creates a fully-connected graph structure with zero-valued edge embeddings.
    Useful for architectural consistency in models that expect graph-structured
    inputs even when edges carry no information.

    Attributes:
        embed_dim (int): Dimensionality of the output zero-embeddings.
        self_loop (bool): Whether to include self-loops in the graph.
    """

    def __init__(self, embed_dim: int, self_loop: bool = False, **kwargs: Any) -> None:
        """Initializes NoEdgeEmbedding.

        Args:
            embed_dim: Resulting edge feature dimension.
            self_loop: Whether to include self-loops in the generated graph.
            kwargs: Unused arguments captured for registry compatibility.
        """
        assert Batch is not None, (
            "torch_geometric is required for NoEdgeEmbedding. Install via: pip install torch_geometric"
        )
        super().__init__()
        self.embed_dim = embed_dim
        self.self_loop = self_loop

    def forward(self, td: Any, init_embeddings: torch.Tensor) -> Batch:
        """Generates zero-valued edge embeddings for a fully-connected graph.

        Args:
            td: Current problem instance data.
            init_embeddings: Static node embeddings of shape (batch, nodes, dim).

        Returns:
            Batch: PyG Batch object with zeroed edge_attr.
        """
        data_list: List[BaseData] = []
        n = init_embeddings.shape[1]
        device = init_embeddings.device
        edge_index = get_full_graph_edge_index(n, self_loop=self.self_loop).to(device)
        m = edge_index.shape[1]

        for node_embed in init_embeddings:
            data = Data(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=torch.zeros((m, self.embed_dim), device=device),
            )
            data_list.append(data)

        return Batch.from_data_list(data_list)
