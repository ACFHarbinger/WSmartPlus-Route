"""Gated Graph Attention Convolution Encoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_gated_convolution_layer import AttentionGatedConvolutionLayer


class GatedGraphAttConvEncoder(nn.Module):
    """
    Encoder stack using AttentionGatedConvolutionLayers.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        n_layers,
        n_sublayers=None,
        feed_forward_hidden=512,
        normalization="batch",
        epsilon_alpha=1e-05,
        learn_affine=True,
        track_stats=False,
        momentum_beta=0.1,
        locresp_k=1.0,
        n_groups=3,
        activation="gelu",
        af_param=1.0,
        threshold=6.0,
        replacement_value=6.0,
        n_params=3,
        uniform_range=[0.125, 1 / 3],
        dropout_rate=0.1,
        agg="sum",
    ):
        """Initializes the GatedGraphAttConvEncoder."""
        super(GatedGraphAttConvEncoder, self).__init__()
        self.embed_dim = embed_dim

        # Initial Edge Embedding (Distance -> Embed)
        self.dist_norm = nn.BatchNorm1d(1)  # Normalize raw distances
        self.init_edge_embed = nn.Linear(1, embed_dim)

        self.layers = nn.ModuleList(
            [
                AttentionGatedConvolutionLayer(
                    n_heads,
                    embed_dim,
                    feed_forward_hidden,
                    normalization,
                    epsilon_alpha,
                    learn_affine,
                    track_stats,
                    momentum_beta,
                    locresp_k,
                    n_groups,
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    uniform_range,
                    gated=True,
                    agg=agg,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges=None, dist=None):
        """Forward pass."""
        if dist is None:
            batch_size, num_nodes, _ = x.size()
            e = torch.zeros(batch_size, num_nodes, num_nodes, self.embed_dim, device=x.device)
        else:
            if len(dist.shape) == 2:
                dist = dist.unsqueeze(0).unsqueeze(-1)
            elif len(dist.shape) == 3:
                dist = dist.unsqueeze(-1)

            B, N1, N2, _ = dist.shape
            dist_flat = dist.view(-1, 1)
            dist_norm = self.dist_norm(dist_flat)
            e = self.init_edge_embed(dist_norm.view(B, N1, N2, 1))

        if edges is None:
            batch_size, num_nodes, _ = x.size()
            edges = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=x.device)

        for layer in self.layers:
            x, e = layer(x, e, mask=edges)

        return self.dropout(x)
