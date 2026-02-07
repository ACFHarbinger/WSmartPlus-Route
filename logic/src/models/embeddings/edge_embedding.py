"""
Edge embedding modules for graph-based encoders.

Transforms raw edge features (e.g., distances between nodes) into learned
embeddings suitable for GNN-based encoders. Supports sparsification to
focus on k-nearest neighbors, which is critical for scaling to large instances.

Contains:
- EdgeEmbedding: Base class for edge embeddings.
- TSPEdgeEmbedding: Distance-based edges for TSP-like problems.
- CVRPEdgeEmbedding: Distance edges with guaranteed depot connectivity.
- NoEdgeEmbedding: Dummy edges for problems without edge features.
- EDGE_EMBEDDING_REGISTRY: Factory registry for dispatching by environment name.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from logic.src.utils.ops import get_distance_matrix, get_full_graph_edge_index, sparsify_graph

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    Batch = Data = None  # type: ignore[assignment, misc]


class EdgeEmbedding(nn.Module):
    """
    Base class for edge embeddings.

    Subclasses define how raw edge features are extracted from problem data
    and projected into the embedding space.
    """

    node_dim: int = 1

    def __init__(
        self,
        embed_dim: int,
        linear_bias: bool = True,
        sparsify: bool = True,
        k_sparse: int | Callable[[int], int] | None = None,
    ):
        """
        Args:
            embed_dim: Output embedding dimension for edge features.
            linear_bias: Whether to include bias in the edge projection.
            sparsify: Whether to create a sparse k-NN graph.
            k_sparse: Number of neighbors per node, or a callable(n) -> k.
                       Defaults to max(n // 5, 10).
        """
        assert (
            Batch is not None
        ), "torch_geometric is required for EdgeEmbedding. Install via: pip install torch_geometric"
        super().__init__()

        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        elif callable(k_sparse):
            self._get_k_sparse = k_sparse
        else:
            raise ValueError("k_sparse must be an int or a callable")

        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td, init_embeddings: torch.Tensor):
        """
        Build a batched PyG graph with embedded edge features.

        Args:
            td: TensorDict or dict with problem data (must contain 'locs').
            init_embeddings: Node embeddings [batch, n_nodes, embed_dim].

        Returns:
            PyG Batch with x, edge_index, and embedded edge_attr.
        """
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch

    def _cost_matrix_to_graph(
        self,
        batch_cost_matrix: torch.Tensor,
        init_embeddings: torch.Tensor,
    ):
        """
        Convert batched cost matrices to a batched PyG graph.

        Args:
            batch_cost_matrix: [batch, n, n] distance/cost matrices.
            init_embeddings: [batch, n, embed_dim] node features.

        Returns:
            PyG Batch object.
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []

        for idx, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(cost_matrix, k_sparse, self_loop=False)
            else:
                edge_index = get_full_graph_edge_index(cost_matrix.shape[0], self_loop=False).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(x=init_embeddings[idx], edge_index=edge_index, edge_attr=edge_attr)
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch


class TSPEdgeEmbedding(EdgeEmbedding):
    """
    Edge embedding for TSP-like problems.

    Uses pairwise Euclidean distances as edge features, optionally
    sparsified to k-nearest neighbors per node.
    """

    node_dim = 1


class CVRPEdgeEmbedding(EdgeEmbedding):
    """
    Edge embedding for capacitated VRP problems.

    Like TSPEdgeEmbedding but ensures all nodes maintain edges to/from
    the depot (node 0), since depot connectivity is critical for VRP feasibility.
    """

    def _cost_matrix_to_graph(
        self,
        batch_cost_matrix: torch.Tensor,
        init_embeddings: torch.Tensor,
    ):
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []

        for idx, cost_matrix in enumerate(batch_cost_matrix):
            n = cost_matrix.shape[0]

            if self.sparsify:
                # Sparsify customer-to-customer edges (exclude depot)
                edge_index, edge_attr = sparsify_graph(cost_matrix[1:, 1:], k_sparse, self_loop=False)
                edge_index = edge_index + 1  # Shift indices to account for removed depot

                # Add depot-to-all and all-to-depot edges
                customer_indices = torch.arange(1, n, device=cost_matrix.device)
                depot_zeros = torch.zeros(n - 1, dtype=torch.long, device=cost_matrix.device)

                depot_edges = torch.cat(
                    [
                        edge_index,
                        torch.stack([customer_indices, depot_zeros]),  # customer -> depot
                        torch.stack([depot_zeros, customer_indices]),  # depot -> customer
                    ],
                    dim=1,
                )
                depot_attrs = torch.cat(
                    [
                        edge_attr,
                        cost_matrix[1:, 0:1],  # customer -> depot distances
                        cost_matrix[0:1, 1:].t(),  # depot -> customer distances
                    ],
                    dim=0,
                )
                edge_index = depot_edges
                edge_attr = depot_attrs
            else:
                edge_index = get_full_graph_edge_index(n, self_loop=False).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(x=init_embeddings[idx], edge_index=edge_index, edge_attr=edge_attr)
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch


class WCVRPEdgeEmbedding(CVRPEdgeEmbedding):
    """
    Edge embedding for Waste Collection VRP problems.

    Inherits from CVRPEdgeEmbedding since WCVRP has the same graph
    structure (depot + customer nodes with capacity constraints).
    """

    pass


class NoEdgeEmbedding(nn.Module):
    """
    Dummy edge embedding for problems that don't use edge features.

    Creates a fully-connected graph with zero-valued edge embeddings.
    Useful for problems where only node features matter (e.g., scheduling).
    """

    def __init__(self, embed_dim: int, self_loop: bool = False, **kwargs):
        assert (
            Batch is not None
        ), "torch_geometric is required for NoEdgeEmbedding. Install via: pip install torch_geometric"
        super().__init__()
        self.embed_dim = embed_dim
        self.self_loop = self_loop

    def forward(self, td, init_embeddings: torch.Tensor):
        data_list = []
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


# =============================================================================
# Registry
# =============================================================================


EDGE_EMBEDDING_REGISTRY = {
    "vrpp": TSPEdgeEmbedding,
    "cvrpp": CVRPEdgeEmbedding,
    "wcvrp": WCVRPEdgeEmbedding,
    "cwcvrp": WCVRPEdgeEmbedding,
    "sdwcvrp": WCVRPEdgeEmbedding,
    "swcvrp": WCVRPEdgeEmbedding,
    "scwcvrp": WCVRPEdgeEmbedding,
    "none": NoEdgeEmbedding,
}


def get_edge_embedding(
    env_name: str,
    embed_dim: int = 128,
    **kwargs,
) -> nn.Module:
    """
    Get problem-specific edge embedding module.

    Args:
        env_name: Environment/problem name.
        embed_dim: Edge embedding dimension.
        **kwargs: Additional arguments passed to the embedding constructor.

    Returns:
        Initialized edge embedding module.
    """
    if env_name not in EDGE_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown environment for edge embedding: {env_name}. Available: {list(EDGE_EMBEDDING_REGISTRY.keys())}"
        )
    return EDGE_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim, **kwargs)


__all__ = [
    "EdgeEmbedding",
    "TSPEdgeEmbedding",
    "CVRPEdgeEmbedding",
    "WCVRPEdgeEmbedding",
    "NoEdgeEmbedding",
    "EDGE_EMBEDDING_REGISTRY",
    "get_edge_embedding",
]
