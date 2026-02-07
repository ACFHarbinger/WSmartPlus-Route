"""
NARGNN Encoder.

GNN-based non-autoregressive encoder with edge heatmap generation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.models.embeddings import get_init_embedding
from logic.src.models.policies.common.nonautoregressive import NonAutoregressiveEncoder

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import BatchNorm
except ImportError:
    Batch = Data = BatchNorm = None  # type: ignore


class EdgeHeatmapGenerator(nn.Module):
    """
    MLP for converting edge embeddings to heatmaps.

    Args:
        embed_dim: Dimension of the embeddings.
        num_layers: Number of linear layers in the network.
        act_fn: Activation function. Defaults to "silu".
        linear_bias: Use bias in linear layers. Defaults to True.
        undirected_graph: Whether the graph is undirected. Defaults to True.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        act_fn: str | Callable = "silu",
        linear_bias: bool = True,
        undirected_graph: bool = True,
    ) -> None:
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=linear_bias) for _ in range(num_layers - 1)])
        self.output = nn.Linear(embed_dim, 1, bias=linear_bias)

        self.act = getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn

        self.undirected_graph = undirected_graph

    def forward(self, graph: Batch) -> torch.Tensor:  # type: ignore
        """Generate heatmap logits from graph edge attributes."""
        # Do not reuse the input value
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr))  # type: ignore

        heatmap_logits = self._make_heatmap_logits(graph)
        return heatmap_logits

    def _make_heatmap_logits(self, batch_graph: Batch) -> torch.Tensor:  # type: ignore
        """Convert batch graph to heatmap logits matrix."""
        graphs = batch_graph.to_data_list() if batch_graph is not None else []
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmap = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmap[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        # Avoid log(0) by adding a small value
        if heatmap.dtype == torch.float32 or heatmap.dtype == torch.bfloat16:
            small_value = 1e-12
        elif heatmap.dtype == torch.float16:
            small_value = 3e-8  # Smallest positive number such that log(small_value) is not -inf
        else:
            raise ValueError(f"Unsupported dtype: {heatmap.dtype}")

        heatmap += small_value
        heatmap_logits = torch.log(heatmap)

        return heatmap_logits


class SimplifiedEdgeEmbedding(nn.Module):
    """Simplified edge embedding for NARGNN using distance matrix."""

    def __init__(self, embed_dim: int, k_sparse: Optional[int] = None, linear_bias: bool = True):
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
        """Create graph from locations."""
        locs = td["locs"]
        batch_size = locs.shape[0]

        # Compute distance matrix
        dist_matrix = torch.cdist(locs, locs, p=2)

        # Build graph for each instance in batch
        graph_data = []
        for i in range(batch_size):
            k = self._get_k_sparse(locs.shape[1])
            # Get k-nearest neighbors for each node
            _, topk_indices = torch.topk(dist_matrix[i], k + 1, dim=-1, largest=False)  # +1 to include self

            # Build edge index
            edge_list = []
            edge_dists = []
            for node_idx in range(locs.shape[1]):
                for neighbor_idx in topk_indices[node_idx]:
                    if neighbor_idx != node_idx:  # Skip self-loops
                        edge_list.append([node_idx, neighbor_idx.item()])
                        edge_dists.append(dist_matrix[i, node_idx, neighbor_idx].item())

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=locs.device).t()
            edge_attr = torch.tensor(edge_dists, dtype=locs.dtype, device=locs.device).unsqueeze(-1)

            graph = Data(x=init_embeddings[i], edge_index=edge_index, edge_attr=edge_attr)  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch


class GNNLayer(nn.Module):
    """Simplified GNN layer for NARGNN."""

    def __init__(self, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        super().__init__()
        assert BatchNorm is not None, "torch_geometric required"

        self.embed_dim = embed_dim
        self.act_fn = getattr(nn.functional, act_fn)

        # Vertex updates
        self.v_lin1 = nn.Linear(embed_dim, embed_dim)
        self.v_lin2 = nn.Linear(embed_dim, embed_dim)
        self.v_lin3 = nn.Linear(embed_dim, embed_dim)
        self.v_lin4 = nn.Linear(embed_dim, embed_dim)
        self.v_bn = BatchNorm(embed_dim)

        # Edge updates
        self.e_lin = nn.Linear(embed_dim, embed_dim)
        self.e_bn = BatchNorm(embed_dim)

        self.agg_fn = agg_fn

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GNN layer."""
        x0, w0 = x, edge_attr

        # Vertex updates with edge-gating
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)

        # Aggregate edge-weighted features
        edge_weighted = torch.sigmoid(w0) * x2[edge_index[1]]
        if self.agg_fn == "mean":
            from torch_scatter import scatter_mean

            aggregated = scatter_mean(edge_weighted, edge_index[0], dim=0, dim_size=x0.size(0))
        else:
            raise NotImplementedError(f"Aggregation {self.agg_fn} not implemented")

        x = x0 + self.act_fn(self.v_bn(x1 + aggregated))

        # Edge updates
        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))

        return x, w


class SimplifiedGNNEncoder(nn.Module):
    """Simplified GNN encoder for NARGNN."""

    def __init__(self, num_layers: int, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        super().__init__()
        self.act_fn = getattr(nn.functional, act_fn)
        self.layers = nn.ModuleList([GNNLayer(embed_dim, act_fn, agg_fn) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process graph through GNN layers."""
        x = self.act_fn(x)
        edge_attr = self.act_fn(edge_attr)
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr


class NARGNNEncoder(NonAutoregressiveEncoder):
    """
    Anisotropic Graph Neural Network encoder with edge-gating mechanism.

    Based on Joshi et al. (2022) and used in DeepACO (Ye et al., 2023).
    Creates a heatmap of NxN for N nodes that models the probability
    to go from one node to another for all nodes.

    Args:
        embed_dim: Dimension of the node embeddings.
        env_name: Name of the environment used to initialize embeddings.
        init_embedding: Model to use for the initial embedding. If None, use default.
        edge_embedding: Model to use for the edge embedding. If None, use default.
        graph_network: Model to use for the graph network. If None, use default.
        heatmap_generator: Model to use for the heatmap generator. If None, use default.
        num_layers_heatmap_generator: Number of layers in the heatmap generator.
        num_layers_graph_encoder: Number of layers in the graph encoder.
        act_fn: Activation function for GNN layers. Defaults to 'silu'.
        agg_fn: Aggregation function for GNN layers. Options: 'add', 'mean', 'max'.
        linear_bias: Use bias in linear layers. Defaults to True.
        k_sparse: Number of edges to keep for each node. Defaults to None.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        act_fn: str = "silu",
        agg_fn: str = "mean",
        linear_bias: bool = True,
        k_sparse: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.env_name = env_name

        self.init_embedding = get_init_embedding(self.env_name, embed_dim) if init_embedding is None else init_embedding

        self.edge_embedding = (
            SimplifiedEdgeEmbedding(embed_dim=embed_dim, k_sparse=k_sparse, linear_bias=linear_bias)
            if edge_embedding is None
            else edge_embedding
        )

        self.graph_network = (
            SimplifiedGNNEncoder(
                embed_dim=embed_dim,
                num_layers=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
            )
            if graph_network is None
            else graph_network
        )

        self.heatmap_generator = (
            EdgeHeatmapGenerator(
                embed_dim=embed_dim,
                num_layers=num_layers_heatmap_generator,
                linear_bias=linear_bias,
            )
            if heatmap_generator is None
            else heatmap_generator
        )

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """
        Forward pass of the encoder.

        Args:
            td: TensorDict with problem data.

        Returns:
            Tuple of (heatmap_logits, node_embed).
        """
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        # Process embedding into graph
        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        # Generate heatmap logits
        heatmap_logits = self.heatmap_generator(graph)

        # Return latent representation (i.e. heatmap logits) and initial embeddings
        return heatmap_logits, node_embed


class NARGNNNodeEncoder(NARGNNEncoder):
    """
    NARGNN encoder variant that returns node embeddings instead of heatmaps.

    This is useful when you want to use the node embeddings directly without
    transforming them into a heatmap.
    """

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning processed node embeddings.

        Args:
            td: TensorDict with problem data.

        Returns:
            Tuple of (proc_embeds, node_embed).
        """
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        # Process embedding into graph
        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        proc_embeds = graph.x
        batch_size = node_embed.shape[0]
        # Reshape proc_embeds from [bs*n, h] to [bs, n, h]
        proc_embeds = proc_embeds.reshape(batch_size, -1, proc_embeds.shape[1])
        return proc_embeds, node_embed
