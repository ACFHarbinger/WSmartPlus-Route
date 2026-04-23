"""__init__.py module.

Attributes:
    MatNetEncoder: Matrix Encoding Network Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.matnet import MatNetEncoder
    >>> encoder = MatNetEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import MatNetEncoder as MatNetEncoder

__all__ = ["MatNetEncoder"]
