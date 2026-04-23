"""Factory pattern for neural components (encoders, decoders).

Attributes:
    NeuralComponentFactory: Abstract base class for all neural component factories.
    AttentionComponentFactory: Factory for attention-based models.
    GCNComponentFactory: Factory for GCN-based models.
    GACComponentFactory: Factory for Graph Attention Convolution models.
    TGCComponentFactory: Factory for Transformer Graph Convolution models.
    GGACComponentFactory: Factory for Gated Graph Attention Convolution models.
    MLPComponentFactory: Factory for MLP-based models.
    MoEComponentFactory: Factory for Mixture of Experts models.
    NARComponentFactory: Factory for Non-Autoregressive models.
    MDAMComponentFactory: Factory for MDAM-based models.
    GFACSComponentFactory: Factory for GFACS-based models.

Example:
    >>> from logic.src.models.subnets.factories import AttentionComponentFactory
    >>> factory = AttentionComponentFactory()
"""

from .attention import AttentionComponentFactory
from .base import NeuralComponentFactory, _create_decoder_by_type
from .gac import GACComponentFactory
from .gcn import GCNComponentFactory
from .gfacs import GFACSComponentFactory
from .ggac import GGACComponentFactory
from .mdam import MDAMComponentFactory
from .mlp import MLPComponentFactory
from .moe import MoEComponentFactory
from .nar import NARComponentFactory
from .tgc import TGCComponentFactory

__all__ = [
    "NeuralComponentFactory",
    "AttentionComponentFactory",
    "GCNComponentFactory",
    "GACComponentFactory",
    "TGCComponentFactory",
    "GGACComponentFactory",
    "MLPComponentFactory",
    "MoEComponentFactory",
    "NARComponentFactory",
    "MDAMComponentFactory",
    "GFACSComponentFactory",
    "_create_decoder_by_type",
]
