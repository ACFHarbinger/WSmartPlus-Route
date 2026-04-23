"""__init__.py module.

Attributes:
    MoEGraphAttentionEncoder: Mixture-of-Experts Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.moe import MoEGraphAttentionEncoder
    >>> encoder = MoEGraphAttentionEncoder(n_layers=3, embed_dim=128)
"""

from .encoder import MoEGraphAttentionEncoder as MoEGraphAttentionEncoder

__all__ = ["MoEGraphAttentionEncoder"]
