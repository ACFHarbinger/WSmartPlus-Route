"""__init__.py module.

Attributes:
    GraphConvolutionLayer: Graph Convolution Layer.
    FFConvSubLayer: Feed-Forward Convolution Sub-Layer.
    TGCFeedForwardSubLayer: Transformer-based Graph Convolution Feed-Forward Sub-Layer.
    TGCMultiHeadAttentionLayer: Transformer-based Graph Convolution Multi-Head Attention Layer.
    TransGraphConvEncoder: Transformer-based Graph Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.tgc import TransGraphConvEncoder
    >>> encoder = TransGraphConvEncoder(n_layers=3, embed_dim=128)
"""

from .conv_layer import GraphConvolutionLayer
from .conv_sublayer import FFConvSubLayer
from .encoder import TransGraphConvEncoder
from .ff_sublayer import TGCFeedForwardSubLayer
from .mha_layer import TGCMultiHeadAttentionLayer

__all__ = [
    "GraphConvolutionLayer",
    "FFConvSubLayer",
    "TGCFeedForwardSubLayer",
    "TGCMultiHeadAttentionLayer",
    "TransGraphConvEncoder",
]
