"""Transformer Graph Convolution Encoder."""

import torch.nn as nn
from logic.src.models.modules import (
    ActivationFunction,
    FeedForward,
    GraphConvolution,
    MultiHeadAttention,
    Normalization,
    SkipConnection,
)


class TGCFeedForwardSubLayer(nn.Module):
    """
    Feed-Forward Sub-Layer with activation.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
    ):
        """Initializes the TGCFeedForwardSubLayer."""
        super(TGCFeedForwardSubLayer, self).__init__()
        self.sub_layers = (
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(
                    activation,
                    af_param,
                    threshold,
                    replacement_value,
                    n_params,
                    dist_range,
                ),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
            if feed_forward_hidden > 0
            else FeedForward(embed_dim, embed_dim)
        )

    def forward(self, h, mask=None):
        """Forward pass."""
        return self.sub_layers(h)


class TGCMultiHeadAttentionLayer(nn.Module):
    """
    Multi-Head Attention Layer with Normalization and Feed-Forward.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization,
        epsilon_alpha,
        learn_affine,
        track_stats,
        mbeta,
        lr_k,
        n_groups,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        uniform_range,
    ):
        """Initializes the TGCMultiHeadAttentionLayer."""
        super(TGCMultiHeadAttentionLayer, self).__init__()
        self.att = SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm1 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )
        self.ff = SkipConnection(
            TGCFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
        )
        self.norm2 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, mask=None):
        """Forward pass with skip connections and normalization."""
        h = self.att(h)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)


class FFConvSubLayer(nn.Module):
    """
    Feed-Forward Convolution Sub-Layer.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        agg,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        dist_range,
        bias=True,
    ):
        """Initializes the FFConvSubLayer."""
        super(FFConvSubLayer, self).__init__()
        self.conv = GraphConvolution(embed_dim, feed_forward_hidden, agg)
        self.af = ActivationFunction(activation, af_param, threshold, replacement_value, n_params, dist_range)
        self.ff = FeedForward(feed_forward_hidden, embed_dim, bias=bias)

    def forward(self, h, mask=None):
        """Forward pass."""
        h = self.conv(h, mask)
        h = self.af(h)
        return self.ff(h)


class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolution Layer with Normalization.
    """

    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        agg,
        normalization,
        epsilon_alpha,
        learn_affine,
        track_stats,
        mbeta,
        lr_k,
        n_groups,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        uniform_range,
    ):
        """Initializes the GraphConvolutionLayer."""
        super(GraphConvolutionLayer, self).__init__()
        self.ff_conv = SkipConnection(
            FFConvSubLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
            )
        )
        self.norm = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, mask):
        """Forward pass."""
        h = self.ff_conv(h, mask=mask)
        return self.norm(h)


class TransGraphConvEncoder(nn.Module):
    """
    Encoder stack of MultiHeadAttentionLayers and GraphConvolutionLayers.
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
        agg="mean",
    ):
        """
        Initializes the TransGraphConvEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of attention layers.
            n_sublayers: Number of GCN layers.
            feed_forward_hidden: Hidden dimension.
            normalization: Normalization type.
            epsilon_alpha: Epsilon.
            learn_affine: Learn affine parameters.
            track_stats: Track stats.
            momentum_beta: Momentum.
            locresp_k: K value.
            n_groups: Number of groups.
            activation: Activation function.
            af_param: Activation parameter.
            threshold: Activation threshold.
            replacement_value: Replacement value.
            n_params: Number of parameters.
            uniform_range: Uniform range.
            dropout_rate: Dropout rate.
            agg: Aggregation method.
        """
        super(TransGraphConvEncoder, self).__init__()
        layers = [
            TGCMultiHeadAttentionLayer(
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
            )
            for _ in range(n_layers)
        ] + [
            GraphConvolutionLayer(
                embed_dim,
                feed_forward_hidden,
                agg,
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
            )
            for _ in range(n_sublayers or 0)
        ]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges):
        """
        Forward pass.

        Args:
            x: Input embeddings.
            edges: Adjacency/Mask.

        Returns:
            Encoded embeddings.
        """
        # edge_idx = torch.reshape(edges, (2, edges.size(0), edges.size(2)))
        for layer in self.layers:
            x = layer(x, edges)
        return self.dropout(x)  # (batch_size, graph_size, embed_dim)
