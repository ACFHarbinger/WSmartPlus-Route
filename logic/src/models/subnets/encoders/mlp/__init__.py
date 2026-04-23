"""__init__.py module.

Attributes:
    MLPEncoder: Simple MLP encoder with configurable activation and normalization.

Example:
    >>> from logic.src.models.subnets.encoders.mlp import MLPEncoder
    >>> encoder = MLPEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import MLPEncoder as MLPEncoder

__all__ = ["MLPEncoder"]
