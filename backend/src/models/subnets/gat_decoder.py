import torch
import torch.nn as nn

from ..modules import (
    MultiHeadAttention, FeedForward, 
    Normalization, SkipConnection, ActivationFunction
)


class FeedForwardSubLayer(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden, activation, af_param,
                threshold, replacement_value, n_params, dist_range, bias=True):
        super(FeedForwardSubLayer, self).__init__()
        self.sub_layers = nn.Sequential(
            FeedForward(embed_dim, feed_forward_hidden, bias=bias),
            ActivationFunction(activation, af_param, threshold, replacement_value, n_params, dist_range),
            FeedForward(feed_forward_hidden, embed_dim, bias=bias),
        ) if feed_forward_hidden > 0 else FeedForward(embed_dim, embed_dim)

    def forward(self, h, mask=None):
        return self.sub_layers(h)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden, normalization, 
                epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups,
                activation, af_param, threshold, replacement_value, n_params, uniform_range):
        super(MultiHeadAttentionLayer, self).__init__()
        self.att = SkipConnection(
            MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
        )
        self.norm1 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
        self.ff = SkipConnection(
            FeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param,
                                threshold, replacement_value, n_params, uniform_range)
        )
        self.norm2 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
    
    def forward(self, q, h, mask):
        h = self.att(q, h, mask)
        h = self.norm1(h)
        h = self.ff(h)
        return self.norm2(h)


class GraphAttentionDecoder(nn.Module):
    def __init__(self, 
                n_heads, 
                embed_dim, 
                n_layers, 
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
                dropout_rate=0.1):
        super(GraphAttentionDecoder, self).__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, 
                                    epsilon_alpha, learn_affine, track_stats, momentum_beta, 
                                    locresp_k, n_groups, activation, af_param, threshold, 
                                    replacement_value, n_params, uniform_range) for _ in range(n_layers)
        ])
        self.projection = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        for layer in self.layers:
            h = layer(q, h, mask)
        
        out = self.projection(self.dropout(h))
        return torch.softmax(out, dim=-1)
