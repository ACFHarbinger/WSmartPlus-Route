"""__init__.py module.

Attributes:
    PointerEncoder: Pointer Network Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.ptr import PointerEncoder
    >>> encoder = PointerEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import PointerEncoder

__all__ = ["PointerEncoder"]
