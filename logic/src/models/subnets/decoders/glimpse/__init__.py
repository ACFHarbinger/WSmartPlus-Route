"""glimpse decoder package.

This package provides decoders and attention mechanisms based on the
Multi-Head Attention (MHA) and glimpse logic.

Attributes:
    GlimpseDecoder: Autoregressive decoder using MHA and glimpse.

Example:
    >>> from logic.src.models.subnets.decoders.glimpse import GlimpseDecoder
    >>> decoder = GlimpseDecoder(...)
"""

from .decoder import GlimpseDecoder

__all__: list[str] = ["GlimpseDecoder"]
