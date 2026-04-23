"""
MDAM Decoder package.

This package provides the Multi-Decoder Attention Model (MDAM) implementation,
designed for constructing diverse sets of routing solutions.

Attributes:
    MDAMDecoder: Parallel multi-path decoder for MDAM models.
    MDAMPath: Single constructive decoding path component.

Example:
    >>> from logic.src.models.subnets.decoders.mdam import MDAMDecoder
    >>> decoder = MDAMDecoder(...)
"""

# Cache moved to common.AttentionDecoderCache
# _decode_probs moved to common.select_action
from .decoder import MDAMDecoder as MDAMDecoder
from .path import MDAMPath as MDAMPath

__all__: list[str] = ["MDAMDecoder", "MDAMPath"]
