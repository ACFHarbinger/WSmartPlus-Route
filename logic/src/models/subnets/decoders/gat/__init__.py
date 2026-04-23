"""Graph Attention (GAT) Decoder: Attention-based constructive decoding.

This module provides decoders that use Graph Attention Networks (GAT) to
evaluate next-node probabilities during constructive routing.

Attributes:
    DeepGATDecoder: High-level GAT-based constructive decoder.
    GraphAttentionDecoder: Multi-layer graph attention mechanism for decoding.

Example:
    >>> from logic.src.models.subnets.decoders.gat import DeepGATDecoder
    >>> decoder = DeepGATDecoder(embed_dim=128, hidden_dim=512, n_heads=8, n_layers=3)
"""

from .decoder import DeepGATDecoder as DeepGATDecoder
from .graph_decoder import GraphAttentionDecoder as GraphAttentionDecoder
