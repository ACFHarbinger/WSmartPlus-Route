#import torch
import torch.nn as nn

from ..modules import (
    Normalization, SkipConnection, ActivationFunction,
    MultiHeadAttention, FeedForward, GraphConvolution,
)


class FFConvSubLayer(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden, agg, activation, af_param,
                threshold, replacement_value, n_params, dist_range, bias=True):
        super(FFConvSubLayer, self).__init__()
        self.conv = GraphConvolution(embed_dim, feed_forward_hidden, aggregation=agg, bias=bias)
        self.af = ActivationFunction(activation, af_param, threshold, replacement_value, n_params, dist_range)
        self.ff = FeedForward(feed_forward_hidden, embed_dim, bias=bias)

    def forward(self, h, mask=None):
        h = self.conv(h, mask)
        h = self.af(h)
        return self.ff(h)


class AttentionConvolutionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden, agg, normalization, 
                epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups,
                activation, af_param, threshold, replacement_value, n_params, uniform_range):
        super(AttentionConvolutionLayer, self).__init__()
        self.att = SkipConnection(
            MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
        )
        self.norm1 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
        self.ff_conv = SkipConnection(
            FFConvSubLayer(embed_dim, feed_forward_hidden, agg, activation, af_param,
                            threshold, replacement_value, n_params, uniform_range)
        )
        self.norm2 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
    
    def forward(self, h, mask):
        h = self.att(h)
        h = self.norm1(h)
        h = self.ff_conv(h, mask)
        return self.norm2(h)


class GraphAttConvEncoder(nn.Module):
    def __init__(self, 
                n_heads, 
                embed_dim, 
                n_layers, 
                n_sublayers=None,
                feed_forward_hidden=512,
                normalization='batch',
                epsilon_alpha=1e-05,
                learn_affine=True,
                track_stats=False,
                momentum_beta=0.1,
                locresp_k=1.0,
                n_groups=3,
                activation='gelu',
                af_param=1.0,
                threshold=6.0,
                replacement_value=6.0,
                n_params=3,
                uniform_range=[0.125, 1/3],
                dropout_rate=0.1,
                agg='sum'):
        super(GraphAttConvEncoder, self).__init__()
        self.layers = nn.ModuleList([
            AttentionConvolutionLayer(n_heads, embed_dim, feed_forward_hidden, agg, normalization, 
                                    epsilon_alpha, learn_affine, track_stats, momentum_beta, 
                                    locresp_k, n_groups, activation, af_param, threshold, 
                                    replacement_value, n_params, uniform_range)
        for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges):
        #edge_idx = torch.reshape(edges, (2, edges.size(0), edges.size(2)))
        for layer in self.layers:
            x = layer(x, edges)
        return self.dropout(x) # (batch_size, graph_size, embed_dim)
