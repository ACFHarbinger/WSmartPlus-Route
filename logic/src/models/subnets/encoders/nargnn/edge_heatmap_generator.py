"""Heatmap generator for NARGNN.

Attributes:
    EdgeHeatmapGenerator: MLP for converting edge embeddings to heatmaps.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn.edge_heatmap_generator import EdgeHeatmapGenerator
    >>> gen = EdgeHeatmapGenerator(embed_dim=128, num_layers=3)
"""

from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn

try:
    from torch_geometric.data import Batch
except ImportError:
    Batch = None  # type: ignore


class EdgeHeatmapGenerator(nn.Module):
    """MLP for converting edge embeddings to heatmaps.

    This module processes edge features through a series of linear layers and
    reconstructs a full heatmap tensor from a sparse graph representation.

    Attributes:
        linears (nn.ModuleList): Internal linear layers for feature processing.
        output (nn.Linear): Final projection to a single logit per edge.
        act (Callable): Activation function.
        undirected_graph (bool): Whether the graph is treated as undirected.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        act_fn: Union[str, Callable] = "silu",
        linear_bias: bool = True,
        undirected_graph: bool = True,
    ) -> None:
        """Initializes the EdgeHeatmapGenerator.

        Args:
            embed_dim: Dimension of the edge embeddings.
            num_layers: Total number of linear layers in the MLP.
            act_fn: Activation function or name of a torch.nn.functional function.
            linear_bias: Whether to use bias in linear layers.
            undirected_graph: Whether to assume the graph is undirected.
        """
        super().__init__()

        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=linear_bias) for _ in range(num_layers - 1)])
        self.output = nn.Linear(embed_dim, 1, bias=linear_bias)

        self.act = getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn
        self.undirected_graph = undirected_graph

    def forward(self, graph: Batch) -> torch.Tensor:  # type: ignore
        """Forward pass of the heatmap generator.

        Processes edge features and converts them to a full square heatmap matrix.

        Args:
            graph: PyG Batch object containing edge features.

        Returns:
            torch.Tensor: Heatmap logits of shape (batch, num_nodes, num_nodes).
        """
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr))  # type: ignore

        heatmap_logits = self._make_heatmap_logits(graph)
        return heatmap_logits

    def _make_heatmap_logits(self, batch_graph: Batch) -> torch.Tensor:  # type: ignore
        """Helper to create heatmap logits from a sparse graph batch.

        Args:
            batch_graph: PyG Batch object.

        Returns:
            torch.Tensor: Heatmap logits tensor.

        Raises:
            ValueError: If the edge attribute dtype is not supported.
        """
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

        if heatmap.dtype in (torch.float32, torch.bfloat16):
            small_value = 1e-12
        elif heatmap.dtype == torch.float16:
            small_value = 3e-8
        else:
            raise ValueError(f"Unsupported dtype: {heatmap.dtype}")

        heatmap += small_value
        heatmap_logits = torch.log(heatmap)

        return heatmap_logits
