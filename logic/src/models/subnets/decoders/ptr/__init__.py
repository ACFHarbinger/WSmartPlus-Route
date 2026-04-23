"""
Pointer decoder package.

This package provides the Pointer Network decoder implementation and its
associated attention mechanism.

Attributes:
    PointerDecoder: Standard Pointer Network decoder.
    PointerAttention: Bahdanau-style attention for pointing decoders.

Example:
    >>> from logic.src.models.subnets.decoders.ptr import PointerDecoder
    >>> decoder = PointerDecoder(...)
"""

from .decoder import PointerDecoder as PointerDecoder
from .pointer_attention import PointerAttention as PointerAttention

__all__: list[str] = ["PointerDecoder", "PointerAttention"]
