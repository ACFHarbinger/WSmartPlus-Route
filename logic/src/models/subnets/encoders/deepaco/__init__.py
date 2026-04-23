"""__init__.py module.

Attributes:
    DeepACOEncoder: DeepACO Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.deepaco import DeepACOEncoder
    >>> encoder = DeepACOEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import DeepACOEncoder

__all__ = ["DeepACOEncoder"]
