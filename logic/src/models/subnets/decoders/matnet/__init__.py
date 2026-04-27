"""matnet decoder package.

This package provides decoders specific to MatNet architectures,
handling matrix-aware problem representations.

Attributes:
    MatNetDecoder: Matrix-aware decoder for MatNet models.

Example:
    >>> from logic.src.models.subnets.decoders.matnet import MatNetDecoder
    >>> decoder = MatNetDecoder(...)
"""

from .decoder import MatNetDecoder as MatNetDecoder

__all__ = [
    "MatNetDecoder",
]
