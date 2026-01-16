"""Graph Convolution Encoder."""

import torch
from torch import nn

from ..modules import GatedGraphConvolution


class GraphConvolutionEncoder(nn.Module):
    """
    Encoder based on Gated Graph Convolutions.
    """

    def __init__(
        self,
        n_layers,
        hidden_dim,
        agg="sum",
        norm="layer",
        learn_affine=True,
        track_norm=False,
        gated=True,
        *args,
        **kwargs,
    ):
        """
        Initializes the GCN Encoder.

        Args:
            n_layers: Number of layers.
            hidden_dim: Hidden dimension.
            agg: Aggregation method.
            norm: Normalization type.
            learn_affine: Whether to learn affine parameters.
            track_norm: Whether to track normalization stats.
            gated: Whether to use gated convolutions.
        """
        super(GraphConvolutionEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList(
            [
                GatedGraphConvolution(
                    hidden_dim=hidden_dim,
                    aggregation=agg,
                    norm=norm,
                    learn_affine=learn_affine,
                    gated=gated,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, edges):
        """
        Args:
            x: Input node features (B x V x H)
            edges: Graph adjacency matrices (B x V x V)
        Returns:
            Updated node features (B x V x H)
        """
        # Embed edge features
        edge_embed = self.init_embed_edges(edges.type(torch.long))

        for layer in self.layers:
            x, edge_embed = layer(x, edge_embed, edges)

        return x
