"""Neural network modules and building blocks.

This package contains reusable layers and sub-networks used to build complex
VRP models, including attention mechanisms, graph convolutions, and custom
normalization layers.

Attributes:
    ActivationFunction: Configurable activation factory.
    MultiHeadAttention: Foundation for neural routing encoders.
    MultiHeadCrossAttention: Cross-modal attention mechanism.
    MultiHeadFlashAttention: Memory-efficient attention implementation.
    GraphConvolution: Standard spatial aggregation layers.
    GatedGraphConvolution: Gated edge-weighted aggregation.
    EfficientGraphConvolution: Optimized multi-head graph layer.
    FeedForward: Standard positional-wise transformation.
    MPNNEncoder: Message Passing Neural Network sequence encoder.
    Normalization: Unified wrapper for Layer/Instance/Batch normalization.
    SkipConnection: Residual and hyper-network wrappers.
    DynamicHyperConnection: Adaptive state-dependent skip mechanism.
    StaticHyperConnection: Fixed weight-sharing skip mechanism.

Example:
    >>> from logic.src.models.subnets.modules import MultiHeadAttention
    >>> mha = MultiHeadAttention(embed_dim=128, n_heads=8)
"""

from .activation_function import ActivationFunction as ActivationFunction
from .cross_attention import (
    MultiHeadCrossAttention as MultiHeadCrossAttention,
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
from .skip_connection import SkipConnection as SkipConnection
from .static_hyper_connection import (
    StaticHyperConnection as StaticHyperConnection,
)

__all__ = [
    "ActivationFunction",
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "MultiHeadFlashAttention",
    "GraphConvolution",
    "GatedGraphConvolution",
    "EfficientGraphConvolution",
    "FeedForward",
    "MPNNEncoder",
    "Normalization",
    "SkipConnection",
    "DynamicHyperConnection",
    "StaticHyperConnection",
    "MixedScoreMHA",
    "MultiHeadAttentionMDAM",
    "MessagePassingLayer",
    "PointerAttnMoE",
    "PolyNetAttention",
]
