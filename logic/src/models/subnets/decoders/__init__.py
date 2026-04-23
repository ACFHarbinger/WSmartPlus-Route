"""Decoders package.

This package provides various constructive and non-autoregressive decoder
architectures for combinatorial optimization problems.

Attributes:
    GraphAttentionDecoder: GAT-based constructive decoder.
    GlimpseDecoder: Decoder using multiple glimpse steps.
    MatNetDecoder: Matrix-based attention decoder.
    MDAMDecoder: Parallel multi-path decoder.
    SimpleNARDecoder: Heatmap-based non-autoregressive decoder.
    PolyNetDecoder: Polynomial attention decoder.
    PointerDecoder: Classical recurrent pointer network decoder.
    ACODecoder: Ant Colony Optimization based decoder.

Example:
    >>> from logic.src.models.subnets.decoders import MDAMDecoder
    >>> decoder = MDAMDecoder(...)
"""

from .common import FeedForwardSubLayer
from .deepaco import ACODecoder
from .gat import DeepGATDecoder, GraphAttentionDecoder
from .glimpse.decoder import GlimpseDecoder
from .matnet import MatNetDecoder
from .mdam import MDAMDecoder
from .nar import SimpleNARDecoder
from .polynet import PolyNetDecoder
from .ptr import PointerDecoder

__all__: list[str] = [
    "FeedForwardSubLayer",
    "ACODecoder",
    "DeepGATDecoder",
    "GraphAttentionDecoder",
    "GlimpseDecoder",
    "MatNetDecoder",
    "MDAMDecoder",
    "SimpleNARDecoder",
    "PolyNetDecoder",
    "PointerDecoder",
]
