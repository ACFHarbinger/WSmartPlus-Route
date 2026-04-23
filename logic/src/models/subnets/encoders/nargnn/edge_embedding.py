"""Simplified edge embedding for NARGNN.

Attributes:
    SimplifiedEdgeEmbedding: Simplified edge embedding for NARGNN using distance matrix.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn.edge_embedding import SimplifiedEdgeEmbedding
    >>> emb = SimplifiedEdgeEmbedding(embed_dim=128)
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    Batch = Data = None  # type: ignore


class SimplifiedEdgeEmbedding(nn.Module):
    """Simplified edge embedding for NARGNN using distance matrix.

    This module constructs a sparse graph from node locations and generates
    initial edge embeddings based on Euclidean distances.

    Attributes:
        k_sparse (Optional[int]): Fixed number of nearest neighbors for sparsity.
        edge_embed (nn.Linear): Linear projection layer for edge distances.
    """

    def __init__(
        self,
        embed_dim: int,
        k_sparse: Optional[int] = None,
        linear_bias: bool = True,
    ) -> None:
        """Initializes the SimplifiedEdgeEmbedding.

        Args:
            embed_dim: Dimension of the output edge embeddings.
            k_sparse: Fixed number of nearest neighbors. If None, it scales with graph size.
            linear_bias: Whether to use bias in the edge embedding linear layer.

        Raises:
            ValueError: If k_sparse is neither an int nor None.
        """
        super().__init__()
        assert Batch is not None, "torch_geometric required for NARGNN"

        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        else:
            raise ValueError("k_sparse must be an int or None")

        self.k_sparse = k_sparse
        self.edge_embed = nn.Linear(1, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict, init_embeddings: torch.Tensor) -> Batch:  # type: ignore
        """Forward pass of the edge embedding module.

        Constructs a k-nearest neighbor graph and projects distances to embeddings.

        Args:
            td: TensorDict containing "locs" of shape (batch, num_nodes, 2).
            init_embeddings: Initial node embeddings of shape (batch, num_nodes, embed_dim).

        Returns:
            Batch: A PyG Batch object representing the sparse graph.
        """
        locs = td["locs"]
        batch_size = locs.shape[0]
        dist_matrix = torch.cdist(locs, locs, p=2)

        graph_data = []
        for i in range(batch_size):
            k = self._get_k_sparse(locs.shape[1])
            _, topk_indices = torch.topk(dist_matrix[i], k + 1, dim=-1, largest=False)

            edge_list = []
            edge_dists = []
            for node_idx in range(locs.shape[1]):
                for neighbor_idx in topk_indices[node_idx]:
                    if neighbor_idx != node_idx:
                        edge_list.append([node_idx, neighbor_idx.item()])
                        edge_dists.append(dist_matrix[i, node_idx, neighbor_idx].item())

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=locs.device).t()
            edge_attr = torch.tensor(edge_dists, dtype=locs.dtype, device=locs.device).unsqueeze(-1)

            graph = Data(x=init_embeddings[i], edge_index=edge_index, edge_attr=edge_attr)  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch
