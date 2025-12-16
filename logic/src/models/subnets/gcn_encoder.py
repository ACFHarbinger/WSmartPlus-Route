import torch

from torch import nn
from ..modules import GatedGraphConvolution


class GraphConvolutionEncoder(nn.Module):
    """Configurable GCN Encoder
    """
    def __init__(self, n_layers, hidden_dim, agg="sum", norm="layer", 
                 learn_affine=True, track_norm=False, gated=True, *args, **kwargs):
        super(GraphConvolutionEncoder, self).__init__()

        self.init_embed_edges = nn.Embedding(2, hidden_dim)

        self.layers = nn.ModuleList([
            GatedGraphConvolution(hidden_dim=hidden_dim, aggregation=agg, norm=norm, learn_affine=learn_affine, gated=gated)
                for _ in range(n_layers)
        ])

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
