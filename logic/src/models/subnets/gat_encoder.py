import torch.nn as nn

from ..modules import (
    MultiHeadAttention, FeedForward, 
    Normalization, ActivationFunction
)
from ..modules.connections import get_connection_module


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
                activation, af_param, threshold, replacement_value, n_params, uniform_range,
                connection_type='residual', expansion_rate=4):
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.att = get_connection_module(
            connection_type,
            MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim),
            embed_dim,
            expansion_rate
        )
        
        self.norm1 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
        
        self.ff = get_connection_module(
            connection_type,
            FeedForwardSubLayer(embed_dim, feed_forward_hidden, activation, af_param,
                                threshold, replacement_value, n_params, uniform_range),
            embed_dim,
            expansion_rate
        )
        
        self.norm2 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
    
    def forward(self, h, mask):
        h = self.att(h, mask=mask)
        
        # Handle Norm for Hyper-Connections (4D input)
        if h.dim() == 4: # (B, S, D, n)
            # Permute to (B, S, n, D) for Norm(D)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm1(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            h = self.norm1(h)
            
        h = self.ff(h)
        
        if h.dim() == 4:
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm2(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            return self.norm2(h)
            
        return h


class GraphAttentionEncoder(nn.Module):
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
                agg=None,
                connection_type='residual',
                expansion_rate=4):
        super(GraphAttentionEncoder, self).__init__()
        
        self.conn_type = connection_type
        self.expansion_rate = expansion_rate
        
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, 
                                    epsilon_alpha, learn_affine, track_stats, momentum_beta, 
                                    locresp_k, n_groups, activation, af_param, threshold, 
                                    replacement_value, n_params, uniform_range,
                                    connection_type=connection_type, expansion_rate=expansion_rate)
        for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges=None):
        # 1. Expand Input (x -> H) if using Hyper-Connections
        if 'hyper' in self.conn_type:
            # x: (B, S, D) -> H: (B, S, D, n)
            # Repeat input across all streams initially
            H = x.unsqueeze(-1).repeat(1, 1, 1, self.expansion_rate)
            curr = H
        else:
            curr = x

        # 2. Pass through layers
        for layer in self.layers:
            curr = layer(curr, mask=edges)

        # 3. Collapse Output (H -> x) if using Hyper-Connections
        if 'hyper' in self.conn_type:
            # Simple mean pooling to return to standard embedding size
            curr = curr.mean(dim=-1)
            
        return self.dropout(curr) # (batch_size, graph_size, embed_dim)
