"""Common shared components for decoder implementations.

This package contains base classes and utilities shared across multiple decoder
implementations to reduce code duplication and maintain architectural consistency.

Attributes:
    AttentionDecoderCache: Unified cache for attention-based decoders.
    FeedForwardSubLayer: Shared feed-forward sublayer for decoders.
    select_action: Helper for deterministic or greedy action selection.
    select_action_log_prob: Helper that returns both action and log probability.

Example:
    >>> from logic.src.models.subnets.decoders.common import AttentionDecoderCache
    >>> cache = AttentionDecoderCache(node_embeddings=emb, graph_context=ctx)
"""

from .cache import AttentionDecoderCache
from .feed_forward_sublayer import FeedForwardSubLayer
from .selection import select_action, select_action_log_prob

__all__ = [
    "AttentionDecoderCache",
    "FeedForwardSubLayer",
    "select_action",
    "select_action_log_prob",
]
