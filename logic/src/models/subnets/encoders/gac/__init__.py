"""__init__.py module.

Attributes:
    GraphAttConvEncoder: Graph Attention Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gac import GraphAttConvEncoder
    >>> encoder = GraphAttConvEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import GraphAttConvEncoder as GraphAttConvEncoder

__all__ = ["GraphAttConvEncoder"]
