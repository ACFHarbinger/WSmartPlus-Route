"""Graph Convolution Encoder."""

import torch
from torch import nn

from logic.src.models.subnets.modules import GatedGraphConvolution


class GraphConvolutionEncoder(nn.Module):
    """
    Encoder based on Gated Graph Convolutions.
    """

    def __init__(
        self,
        n_layers,
        feed_forward_hidden,
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
            feed_forward_hidden: Hidden dimension.
            agg: Aggregation method.
            norm: Normalization type.
            learn_affine: Whether to learn affine parameters.
            track_norm: Whether to track normalization stats.
            gated: Whether to use gated convolutions.
        """
        super(GraphConvolutionEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, feed_forward_hidden)

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

    def forward(self, x, edges):
        """
        Args:
            x: Input node features (B x V x H)
            edges: Graph adjacency matrices (B x V x V)
        Returns:
            Updated node features (B x V x H)
        """
        batch_size, num_nodes, _ = x.size()

        # Convert sparse [B, 2, E] to dense [B, V, V] if needed
        if edges.dim() == 3 and edges.size(1) == 2:
            adj = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device, dtype=torch.long)
            # Assuming edges[b] are indices.
            # We can scatter.
            for b in range(batch_size):
                # Ensure indices are long
                idx = edges[b].long()
                # Check for out of bounds just in case
                valid_mask = (idx[0] < num_nodes) & (idx[1] < num_nodes)
                src = idx[0][valid_mask]
                dst = idx[1][valid_mask]
                adj[b, src, dst] = 1
            edges = adj

        # Embed edge features
        edge_embed = self.init_embed_edges(edges.type(torch.long))

        for layer in self.layers:
            x, edge_embed = layer(x, edge_embed, edges)

        return x
