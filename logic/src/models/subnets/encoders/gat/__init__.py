"""Graph Attention Encoder implementation.

Attributes:
    GraphAttentionEncoder: Graph Attention Encoder with stacked MultiHeadAttentionLayers.
    GATFeedForwardSubLayer: Feed-Forward Sub-Layer for GAT Encoder.
    GATMultiHeadAttentionLayer: Multi-Head Attention Layer for GAT Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
    >>> encoder = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=3)
"""

from .encoder import GraphAttentionEncoder
from .gat_feed_forward_sublayer import GATFeedForwardSubLayer
from .gat_multi_head_attention_layer import GATMultiHeadAttentionLayer

__all__ = ["GraphAttentionEncoder", "GATFeedForwardSubLayer", "GATMultiHeadAttentionLayer"]
