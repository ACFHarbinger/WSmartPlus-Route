"""__init__.py module.

Attributes:
    MDAMGraphAttentionEncoder: Multi-Decoder Attention Model Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.mdam import MDAMGraphAttentionEncoder
    >>> encoder = MDAMGraphAttentionEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import MDAMGraphAttentionEncoder as MDAMGraphAttentionEncoder

__all__ = ["MDAMGraphAttentionEncoder"]
