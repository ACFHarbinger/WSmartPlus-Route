"""Graph Convolution Encoder.

Attributes:
    GraphConvolutionEncoder: Encoder based on Gated Graph Convolutions.

Example:
    >>> from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder
    >>> encoder = GraphConvolutionEncoder(n_layers=3, feed_forward_hidden=128)
"""

from typing import Any

import torch
from torch import nn

from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules import GatedGraphConvolution


class GraphConvolutionEncoder(nn.Module):
    """Encoder based on Gated Graph Convolutions.

    Uses stacked GatedGraphConvolution layers for node and edge feature updates.
    Unlike transformer-based encoders, this encoder operates on graph structure
    directly using graph convolution operations.

    Attributes:
        norm_config (NormalizationConfig): Normalization configuration.
        n_layers (int): Number of graph convolution layers.
        hidden_dim (int): Hidden dimension for node and edge features.
        agg (str): Aggregation method for neighbor features.
        gated (bool): Whether to use gated update mechanism.
        init_embed_edges (nn.Embedding): Initial edge embedding layer.
        layers (nn.ModuleList): Stack of gated graph convolution layers.
    """

    def __init__(
        self,
        n_layers: int,
        feed_forward_hidden: int,
        agg: str = "sum",
        norm: str = "layer",
        learn_affine: bool = True,
        track_norm: bool = False,
        gated: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the GraphConvolutionEncoder.

        Args:
            n_layers: Number of graph convolution layers.
            feed_forward_hidden: Hidden dimension size for node and edge features.
            agg: Aggregation method: "sum", "mean", or "max".
            norm: Normalization type: "batch", "layer", "instance", or "group".
            learn_affine: Whether to learn affine parameters in normalization.
            track_norm: Whether to track running statistics in normalization.
            gated: Whether to use gated mechanism in graph convolutions.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
        """
        super(GraphConvolutionEncoder, self).__init__()

        # Create normalization config for consistency with other encoders
        self.norm_config = NormalizationConfig(
            norm_type=norm,
            learn_affine=learn_affine,
            track_stats=track_norm,
        )

        # Store layer configuration
        self.n_layers = n_layers
        self.hidden_dim = feed_forward_hidden
        self.agg = agg
        self.gated = gated

        # Edge embedding: binary edge features (0/1) -> hidden_dim
        self.init_embed_edges = nn.Embedding(2, feed_forward_hidden)

        # Stack of gated graph convolution layers
        self.layers = nn.ModuleList(
            [
                GatedGraphConvolution(
                    hidden_dim=feed_forward_hidden,
                    aggregation=agg,
                    norm=norm,
                    learn_affine=learn_affine,
                    gated=gated,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph convolution layers.

        Args:
            x: Input node features of shape (batch_size, num_nodes, hidden_dim).
            edges: Graph adjacency information. Can be dense (batch_size, num_nodes,
                num_nodes) or sparse (batch_size, 2, num_edges).

        Returns:
            torch.Tensor: Updated node features of shape (batch_size, num_nodes, hidden_dim).

        Note:
            If edges are provided in sparse format (edge index list), they are
            converted to dense adjacency matrices before processing.
        """
        batch_size, num_nodes, _ = x.size()

        # Convert sparse edge list to dense adjacency matrix if needed
        if edges.dim() == 3 and edges.size(1) == 2:
            adj = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device, dtype=torch.long)
            # Build adjacency matrix from edge indices
            for b in range(batch_size):
                # Ensure indices are long integers
                idx = edges[b].long()
                # Filter out-of-bounds indices
                valid_mask = (idx[0] < num_nodes) & (idx[1] < num_nodes)
                src = idx[0][valid_mask]
                dst = idx[1][valid_mask]
                adj[b, src, dst] = 1
            edges = adj

        # Embed edge features: {0, 1} -> hidden_dim
        edge_embed = self.init_embed_edges(edges.type(torch.long))

        # Pass through graph convolution layers
        for layer in self.layers:
            x, edge_embed = layer(x, edge_embed, edges)

        return x
