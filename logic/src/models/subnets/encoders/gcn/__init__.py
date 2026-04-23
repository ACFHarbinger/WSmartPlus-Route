"""__init__.py module.

Attributes:
    GraphConvolutionEncoder: Graph Convolution Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder
    >>> encoder = GraphConvolutionEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import GraphConvolutionEncoder

__all__ = ["GraphConvolutionEncoder"]
