"""
Auxiliary layers for NARGNN Encoder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import BatchNorm
except ImportError:
    Batch = Data = BatchNorm = None  # type: ignore


class EdgeHeatmapGenerator(nn.Module):
    """
    MLP for converting edge embeddings to heatmaps.
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
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr))  # type: ignore

        heatmap_logits = self._make_heatmap_logits(graph)
        return heatmap_logits

    def _make_heatmap_logits(self, batch_graph: Batch) -> torch.Tensor:  # type: ignore
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

        if heatmap.dtype == torch.float32 or heatmap.dtype == torch.bfloat16:
            small_value = 1e-12
        elif heatmap.dtype == torch.float16:
            small_value = 3e-8
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


class GNNLayer(nn.Module):
    """Simplified GNN layer for NARGNN."""

    def __init__(self, embed_dim: int, act_fn: str = "silu", agg_fn: str = "mean"):
        super().__init__()
        assert BatchNorm is not None, "torch_geometric required"

        self.embed_dim = embed_dim
        self.act_fn = getattr(nn.functional, act_fn)

        self.v_lin1 = nn.Linear(embed_dim, embed_dim)
        self.v_lin2 = nn.Linear(embed_dim, embed_dim)
        self.v_lin3 = nn.Linear(embed_dim, embed_dim)
        self.v_lin4 = nn.Linear(embed_dim, embed_dim)
        self.v_bn = BatchNorm(embed_dim)

        self.e_lin = nn.Linear(embed_dim, embed_dim)
        self.e_bn = BatchNorm(embed_dim)

        self.agg_fn = agg_fn

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, w0 = x, edge_attr

        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)

        edge_weighted = torch.sigmoid(w0) * x2[edge_index[1]]
        if self.agg_fn == "mean":
            from torch_scatter import scatter_mean

            aggregated = scatter_mean(edge_weighted, edge_index[0], dim=0, dim_size=x0.size(0))
        else:
            raise NotImplementedError(f"Aggregation {self.agg_fn} not implemented")

        x = x0 + self.act_fn(self.v_bn(x1 + aggregated))

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
        x = self.act_fn(x)
        edge_attr = self.act_fn(edge_attr)
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
