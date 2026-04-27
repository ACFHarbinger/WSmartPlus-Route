"""Sub-networks (Encoders, Decoders, Predictors) for neural models.

This package provides various building blocks for neural combinatorial optimization
models, including different flavors of encoders (Graph Attention, MatNet, MLP) and
decoders (Pointer, MDAM).

Attributes:
    GraphAttentionEncoder: Standard GAT-based encoder.
    PointerDecoder: Classic pointer network decoder.
    GraphAttentionDecoder: GAT-based decoder.

Example:
    >>> from logic.src.models.subnets import GraphAttentionEncoder, PointerDecoder
    >>> encoder = GraphAttentionEncoder(node_dim=2, embed_dim=128)
    >>> decoder = PointerDecoder(embed_dim=128)
"""

from .decoders.gat import GraphAttentionDecoder as GraphAttentionDecoder
from .decoders.mdam import MDAMDecoder as MDAMDecoder
from .decoders.polynet import PolyNetDecoder as PolyNetDecoder
from .decoders.ptr import (
    PointerAttention as PointerAttention,
)
from .decoders.ptr import (
    PointerDecoder as PointerDecoder,
)
from .encoders import (
    DeepACOEncoder as DeepACOEncoder,
)
from .encoders import (
    GatedGraphAttConvEncoder as GatedGraphAttConvEncoder,
)
from .encoders import (
    GFACSEncoder as GFACSEncoder,
)
from .encoders import (
    GraphAttConvEncoder as GraphAttConvEncoder,
)
from .encoders import (
    GraphAttentionEncoder as GraphAttentionEncoder,
)
from .encoders import (
    GraphConvolutionEncoder as GraphConvolutionEncoder,
)
from .encoders import (
    MatNetEncoder as MatNetEncoder,
)
from .encoders import (
    MDAMGraphAttentionEncoder as MDAMGraphAttentionEncoder,
)
from .encoders import (
    MLPEncoder as MLPEncoder,
)
from .encoders import (
    MoEGraphAttentionEncoder as MoEGraphAttentionEncoder,
)
from .encoders import (
    NARGNNEncoder as NARGNNEncoder,
)
from .encoders import (
    PointerEncoder as PointerEncoder,
)
from .encoders import (
    TransGraphConvEncoder as TransGraphConvEncoder,
)
from .other.gru_fill_predictor import GatedRecurrentUnitFillPredictor as GatedRecurrentUnitFillPredictor
from .other.lstm_fill_predictor import LongShortTermMemoryFillPredictor as LongShortTermMemoryFillPredictor

__all__ = [
    "GraphAttentionDecoder",
    "MDAMDecoder",
    "PolyNetDecoder",
    "PointerAttention",
    "PointerDecoder",
    "DeepACOEncoder",
    "GatedGraphAttConvEncoder",
    "GFACSEncoder",
    "GraphAttConvEncoder",
    "GraphAttentionEncoder",
    "GraphConvolutionEncoder",
    "MatNetEncoder",
    "MDAMGraphAttentionEncoder",
    "MLPEncoder",
    "MoEGraphAttentionEncoder",
    "NARGNNEncoder",
    "PointerEncoder",
    "TransGraphConvEncoder",
    "GatedRecurrentUnitFillPredictor",
    "LongShortTermMemoryFillPredictor",
]
