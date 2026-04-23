"""Common shared components for encoder implementations.

This package contains base classes and utilities shared across multiple encoder
implementations to reduce code duplication.

Attributes:
    TransformerEncoderBase: Abstract base class for transformer-style graph encoders.
    EncoderFeedForwardSubLayer: Reusable Feed-Forward Sub-Layer.
    MultiHeadAttentionLayerBase: Base class for multi-head attention layers.

Example:
    >>> from logic.src.models.subnets.encoders.common import EncoderFeedForwardSubLayer
    >>> layer = EncoderFeedForwardSubLayer(embed_dim=128, feed_forward_hidden=512)
"""

from .encoder_base import TransformerEncoderBase
from .feed_forward_sublayer import EncoderFeedForwardSubLayer
from .multi_head_attention_layer import MultiHeadAttentionLayerBase

__all__ = [
    "TransformerEncoderBase",
    "EncoderFeedForwardSubLayer",
    "MultiHeadAttentionLayerBase",
]
