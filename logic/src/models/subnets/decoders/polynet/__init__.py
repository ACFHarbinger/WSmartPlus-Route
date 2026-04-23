"""
PolyNet Decoder package.

This package provides the PolyNetDecoder, which implements a polynomial-based
attention mechanism for constructive routing.

Attributes:
    PolyNetDecoder: Decoder utilizing polynomial attention for diversity.

Example:
    >>> from logic.src.models.subnets.decoders.polynet import PolyNetDecoder
    >>> decoder = PolyNetDecoder(k=3, ...)
"""

# Cache moved to common.AttentionDecoderCache
from .decoder import PolyNetDecoder as PolyNetDecoder

__all__: list[str] = ["PolyNetDecoder"]
