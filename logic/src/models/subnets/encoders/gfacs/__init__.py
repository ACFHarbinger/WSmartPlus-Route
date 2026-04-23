"""__init__.py module.

Attributes:
    GFACSEncoder: Graph Feature-Aware Convolutional Sequence Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gfacs import GFACSEncoder
    >>> encoder = GFACSEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import GFACSEncoder

__all__ = ["GFACSEncoder"]
