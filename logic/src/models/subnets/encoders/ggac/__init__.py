"""__init__.py module.

Attributes:
    GatedGraphAttConvEncoder: Gated Graph Attention Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.ggac import GatedGraphAttConvEncoder
    >>> encoder = GatedGraphAttConvEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import GatedGraphAttConvEncoder as GatedGraphAttConvEncoder

__all__ = ["GatedGraphAttConvEncoder"]
