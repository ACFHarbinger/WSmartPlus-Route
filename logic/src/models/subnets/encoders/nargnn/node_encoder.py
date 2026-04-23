"""NARGNN Node Encoder variant.

Attributes:
    NARGNNNodeEncoder: NARGNN encoder variant that returns node embeddings instead of heatmaps.

Example:
    >>> from logic.src.models.subnets.encoders.nargnn.node_encoder import NARGNNNodeEncoder
    >>> enc = NARGNNNodeEncoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Tuple

import torch
from tensordict import TensorDict

from .encoder import NARGNNEncoder


class NARGNNNodeEncoder(NARGNNEncoder):
    """NARGNN encoder variant that returns node embeddings instead of heatmaps.

    This variant focuses on producing refined node embeddings after graph processing,
    which can be used by downstream autoregressive decoders.

    Attributes:
        env_name (str): Name of the environment.
        init_embedding (nn.Module): Initial node embedding layer.
        edge_embedding (nn.Module): Initial edge embedding layer.
        graph_network (nn.Module): GNN encoder for processing node and edge features.
        heatmap_generator (nn.Module): Module for generating edge logits (inherited but unused here).
    """

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Forward pass of the node encoder.

        Processes the graph and returns refined node embeddings.

        Args:
            td: TensorDict containing problem instance data.

        Returns:
            Tuple containing:
                - proc_embeds (torch.Tensor): Processed node embeddings of shape (batch, num_nodes, embed_dim).
                - node_embed (torch.Tensor): Initial node embeddings.
        """
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        graph.x, graph.edge_attr = self.graph_network(graph.x, graph.edge_index, graph.edge_attr)

        proc_embeds = graph.x
        batch_size = node_embed.shape[0]
        proc_embeds = proc_embeds.reshape(batch_size, -1, proc_embeds.shape[1])
        return proc_embeds, node_embed
