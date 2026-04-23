"""Encoder Subnetworks.

Attributes:
    TransformerEncoderBase: Abstract base class for transformer-style graph encoders.
    EncoderFeedForwardSubLayer: Reusable Feed-Forward Sub-Layer.
    MultiHeadAttentionLayerBase: Base class for multi-head attention layers.
    DeepACOEncoder: DeepACO-style graph encoder.
    GraphAttConvEncoder: Graph Attention Convolution Encoder.
    GraphAttentionEncoder: Standard Graph Attention Encoder.
    GraphConvolutionEncoder: Gated Graph Convolution Encoder.
    GFACSEncoder: GFlowNet-ACO Heatmap Predictor Encoder.
    GatedGraphAttConvEncoder: Gated Graph Attention Convolution Encoder.
    MatNetEncoder: Matrix Encoding Network Encoder.
    MDAMGraphAttentionEncoder: Multi-Decoder Attention Model Encoder.
    MLPEncoder: Multilayer Perceptron Encoder.
    MoEGraphAttentionEncoder: Mixture-of-Experts Graph Attention Encoder.
    NARGNNEncoder: Non-Autoregressive Graph Neural Network Encoder.
    NARGNNNodeEncoder: Node-only NARGNN Encoder.
    PointerEncoder: Pointer Network Encoder.
    TransGraphConvEncoder: Transformer-based Graph Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders import MLPEncoder
    >>> encoder = MLPEncoder(embed_dim=128, n_layers=3)
"""

from .common import (
    EncoderFeedForwardSubLayer,
    MultiHeadAttentionLayerBase,
    TransformerEncoderBase,
)
from .deepaco import DeepACOEncoder
from .gac import GraphAttConvEncoder
from .gat import GraphAttentionEncoder
from .gcn import GraphConvolutionEncoder
from .gfacs import GFACSEncoder
from .ggac import GatedGraphAttConvEncoder
from .matnet import MatNetEncoder
from .mdam import MDAMGraphAttentionEncoder
from .mlp import MLPEncoder
from .moe import MoEGraphAttentionEncoder
from .nargnn import NARGNNEncoder, NARGNNNodeEncoder
from .ptr import PointerEncoder
from .tgc import TransGraphConvEncoder

__all__ = [
    # Base classes and common components
    "TransformerEncoderBase",
    "EncoderFeedForwardSubLayer",
    "MultiHeadAttentionLayerBase",
    # Encoder implementations
    "DeepACOEncoder",
    "GraphAttConvEncoder",
    "GraphAttentionEncoder",
    "GraphConvolutionEncoder",
    "GFACSEncoder",
    "GatedGraphAttConvEncoder",
    "MatNetEncoder",
    "MDAMGraphAttentionEncoder",
    "MLPEncoder",
    "MoEGraphAttentionEncoder",
    "NARGNNEncoder",
    "NARGNNNodeEncoder",
    "PointerEncoder",
    "TransGraphConvEncoder",
]
