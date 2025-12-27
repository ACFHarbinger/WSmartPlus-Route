
import torch
import torch.nn as nn
from ..modules import (
    MultiHeadAttention, FeedForward, 
    Normalization, SkipConnection, ActivationFunction,
    GatedGraphConvolution
)

class AttentionGatedConvolutionLayer(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden, normalization, 
                 epsilon_alpha, learn_affine, track_stats, mbeta, lr_k, n_groups,
                 activation, af_param, threshold, replacement_value, n_params, uniform_range,
                 gated=True, agg='sum', bias=True):
        super(AttentionGatedConvolutionLayer, self).__init__()
        
        # 1. Multi-Head Attention (Global Context)
        self.att = SkipConnection(
            MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
        )
        self.norm1 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)
        
        # 2. Gated Graph Convolution (Local Structure + Edge Updates)
        # Note: GatedGraphConvolution takes (h, e, mask) and returns (h, e)
        # We need to wrap it to fit SkipConnection if we want to skip h
        self.gated_gcn = GatedGraphConvolution(
            hidden_dim=embed_dim,
            aggregation=agg,
            norm=normalization,
            activation=activation,
            learn_affine=learn_affine,
            gated=gated,
            bias=bias
        )
        # We'll handle skip on h manually inside forward
        self.norm2 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)

        # 3. Feed Forward (Node-wise processing)
        self.ff = SkipConnection(
            nn.Sequential(
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                ActivationFunction(activation, af_param, threshold, replacement_value, n_params, uniform_range),
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        ) if feed_forward_hidden > 0 else SkipConnection(FeedForward(embed_dim, embed_dim))
        
        self.norm3 = Normalization(embed_dim, normalization, epsilon_alpha, 
                                learn_affine, track_stats, mbeta, n_groups, lr_k)

    def forward(self, h, e, mask=None):
        # 1. MHA
        h = self.att(h, mask=mask)
        h = self.norm1(h)
        
        # 2. Gated GCN
        # GatedGCN returns h_new, e_new. It does residual internally?
        # Checked code: logic/src/models/modules/gated_graph_convolution.py
        # It has "Make residual connection" commented out line 94. 
        # So we should validly add residual here for h.
        # But e is also updated. 
        
        h_in = h
        e_in = e
        
        h_gcn, e_gcn = self.gated_gcn(h, e, mask)
        
        h = h_in + h_gcn # Residual for Node
        e = e_in + e_gcn # Residual for Edge - Wait, lines 73 update e completely. Usually edge residual is good.
        
        h = self.norm2(h)
        # e normalization happens inside GatedGCN if enabled, or we can add it here.
        # GatedGCN code has norm commented out. We might want to trust the module or add it.
        # For now, simplistic.

        # 3. FF
        h = self.ff(h)
        h = self.norm3(h)
        
        return h, e

class GatedGraphAttConvEncoder(nn.Module):
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
        super(GatedGraphAttConvEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Initial Edge Embedding (Distance -> Embed)
        self.dist_norm = nn.BatchNorm1d(1) # Normalize raw distances
        self.init_edge_embed = nn.Linear(1, embed_dim) 

        self.layers = nn.ModuleList([
            AttentionGatedConvolutionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, 
                                        epsilon_alpha, learn_affine, track_stats, momentum_beta, 
                                        locresp_k, n_groups, activation, af_param, threshold, 
                                        replacement_value, n_params, uniform_range,
                                        gated=True, agg=agg)
        for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edges=None, dist=None):
        """
        x: (B, N, D)
        edges: (B, N, N) boolean mask (optional)
        dist: (B, N, N) distance matrix
        """
        if dist is None:
            # Fallback if no dist provided (should warn)
            batch_size, num_nodes, _ = x.size()
            e = torch.zeros(batch_size, num_nodes, num_nodes, self.embed_dim, device=x.device)
        else:
            # Expand dist to (B, N, N, 1) and map to embed_dim
            if len(dist.shape) == 2:
                # Add Batch and Feature dims: (N, N) -> (1, N, N, 1)
                dist = dist.unsqueeze(0).unsqueeze(-1)
            elif len(dist.shape) == 3:
                # Add Feature dim: (B, N, N) -> (B, N, N, 1)
                dist = dist.unsqueeze(-1)
            
            # Normalize dist: (B, N, N, 1) -> (B*N*N, 1) -> BN*N, 1 -> B, N, N, 1
            B, N1, N2, _ = dist.shape
            dist_flat = dist.view(-1, 1)
            dist_norm = self.dist_norm(dist_flat)
            e = self.init_edge_embed(dist_norm.view(B, N1, N2, 1)) 
            
        # Ensure mask is correct format for GatedGCN (0 for edge, 1 for no edge? or Adjacency?)
        # GatedGCN doc: "mask: Graph adjacency matrices (B x V x V)"
        # GatedGCN aggregate: Vh[mask.unsqueeze(-1)...] = 0
        # This implies mask=1 means "No Edge" (Masked out).
        
        # AttentionModel passes 'edges' as `mask` (True = Masked/Skip).
        # So we can pass `edges` directly as mask.
        
        if edges is None:
            batch_size, num_nodes, _ = x.size()
            # Create empty mask (all visible) - False
            edges = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=x.device)

        for layer in self.layers:
            x, e = layer(x, e, mask=edges)
            
        return self.dropout(x)
