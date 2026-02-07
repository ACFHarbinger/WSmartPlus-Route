"""
Encoder Subnetworks.
"""

from .deepaco import DeepACOEncoder
from .gac import GraphAttConvEncoder
from .gat import GraphAttentionEncoder
from .gcn import GraphConvolutionEncoder
from .gfacs import GFACSEncoder
from .ggac import GatedGraphAttConvEncoder
from .ham import HAMEncoder
from .l2d import L2DEncoder
from .matnet import MatNetEncoder
from .mdam import MDAMGraphAttentionEncoder
from .mlp import MLPEncoder
from .moe import MoEGraphAttentionEncoder
from .nargnn import NARGNNEncoder, NARGNNNodeEncoder
from .ptr import PointerEncoder
from .tgc import TransGraphConvEncoder

__all__ = [
    "DeepACOEncoder",
    "GraphAttConvEncoder",
    "GraphAttentionEncoder",
    "GraphConvolutionEncoder",
    "GFACSEncoder",
    "GatedGraphAttConvEncoder",
    "HAMEncoder",
    "L2DEncoder",
    "MatNetEncoder",
    "MDAMGraphAttentionEncoder",
    "MLPEncoder",
    "MoEGraphAttentionEncoder",
    "NARGNNEncoder",
    "NARGNNNodeEncoder",
    "PointerEncoder",
    "TransGraphConvEncoder",
]
