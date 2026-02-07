"""
Neural network modules and building blocks.

This package contains reusable layers and sub-networks used to build complex VRP models.

Attention Mechanisms:
-   `MultiHeadAttention`: Standard MHA.
-   `MultiHeadFlashAttention`: FlashAttention-based MHA (renamed to `flash_attention.py`).
-   `MultiHeadCrossAttention`: Cross-attention between two sequences.
-   `MixedScoreMHA` (MatNet): Attention with mixed scoring functions.
-   `MultiHeadAttentionMDAM`: MDAM-specific attention.

Graph Convolutions:
-   `GraphConvolution`: Basic GCN.
-   `GatedGraphConvolution`: Gated GCN (Residual).
-   `EfficientGraphConvolution`: Memory-efficient GCN.
-   `MessagePassingLayer`: General message passing (MPNN).

Common Components:
-   `Normalization`: Unified interface for Batch/Layer/Instance norm.
-   `FeedForward`: MLP blocks.
-   `SkipConnection`: Residual connections.
-   `ActivationFunction`: Configurable activations.
"""

from .absolute_positional_embedding import (
    AbsolutePositionalEmbedding as AbsolutePositionalEmbedding,
)
from .activation_function import ActivationFunction as ActivationFunction
from .cross_attention import (
    MultiHeadCrossAttention as MultiHeadCrossAttention,
)
from .cyclic_positional_embedding import (
    CyclicPositionalEmbedding as CyclicPositionalEmbedding,
)
from .dynamic_hyper_connection import (
    DynamicHyperConnection as DynamicHyperConnection,
)
from .efficient_graph_convolution import (
    EfficientGraphConvolution as EfficientGraphConvolution,
)
from .feed_forward import FeedForward as FeedForward
from .flash_attention import (
    MultiHeadFlashAttention as MultiHeadFlashAttention,
)
from .gated_graph_convolution import GatedGraphConvolution as GatedGraphConvolution
from .graph_convolution import GraphConvolution as GraphConvolution
from .hgnn import HetGNNLayer as HetGNNLayer
from .matnet_attention import MixedScoreMHA as MixedScoreMHA
from .mdam_attention import (
    MultiHeadAttentionMDAM as MultiHeadAttentionMDAM,
)
from .mpnn_encoder import MPNNEncoder as MPNNEncoder
from .mpnn_layer import MessagePassingLayer as MessagePassingLayer
from .multi_head_attention import MultiHeadAttention as MultiHeadAttention
from .normalization import Normalization as Normalization
from .pointer_attn_moe import PointerAttnMoE
from .polynet_attention import PolyNetAttention as PolyNetAttention
from .positional_embedding_factory import pos_init_embedding as pos_init_embedding
from .skip_connection import SkipConnection as SkipConnection
from .static_hyper_connection import (
    StaticHyperConnection as StaticHyperConnection,
)
