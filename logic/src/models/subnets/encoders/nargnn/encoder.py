"""NARGNN Encoder base implementation.

Attributes:
    NARGNNEncoder: Anisotropic Graph Neural Network encoder with edge-gating mechanism.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder
    >>> encoder = NARGNNEncoder(embed_dim=128, env_name="tsp")
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.non_autoregressive.encoder import NonAutoregressiveEncoder
from logic.src.models.subnets.embeddings import get_init_embedding

from .edge_embedding import SimplifiedEdgeEmbedding
from .edge_heatmap_generator import EdgeHeatmapGenerator
from .gnn_encoder import SimplifiedGNNEncoder


class NARGNNEncoder(NonAutoregressiveEncoder):
    """Anisotropic Graph Neural Network encoder with edge-gating mechanism.

    This encoder processes graph structured data to generate node embeddings and
    edge heatmap logits for non-autoregressive solution construction.

    Attributes:
        env_name (str): Name of the environment (e.g., "tsp", "vrp").
        init_embedding (nn.Module): Initial node embedding layer.
        edge_embedding (nn.Module): Initial edge embedding layer.
        graph_network (nn.Module): GNN encoder for processing node and edge features.
        heatmap_generator (nn.Module): Module for generating edge logits from embeddings.
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
        **kwargs: Any,
    ) -> None:
        """Initializes the NARGNNEncoder.

        Args:
            embed_dim: Dimension of node and edge embeddings.
            env_name: Name of the problem environment.
            init_embedding: Optional custom initial embedding module.
            edge_embedding: Optional custom edge embedding module.
            graph_network: Optional custom GNN encoder module.
            heatmap_generator: Optional custom heatmap generator module.
            num_layers_heatmap_generator: Number of layers in the heatmap generator.
            num_layers_graph_encoder: Number of layers in the GNN encoder.
            act_fn: Activation function used within GNN layers.
            agg_fn: Aggregation function for GNN message passing.
            linear_bias: Whether to use bias in linear layers.
            k_sparse: Number of nearest neighbors for sparse edge construction.
            kwargs: Additional keyword arguments.
        """
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
        """Forward pass of the NARGNN encoder.

        Processes the input TensorDict to produce edge heatmap logits and node embeddings.

        Args:
            td: TensorDict containing problem instance data.

        Returns:
            Tuple containing:
                - heatmap_logits (torch.Tensor): Logits for edge selection.
                - node_embed (torch.Tensor): Final node embeddings.
        """
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        heatmap_logits = self.heatmap_generator(graph)
        return heatmap_logits, node_embed
