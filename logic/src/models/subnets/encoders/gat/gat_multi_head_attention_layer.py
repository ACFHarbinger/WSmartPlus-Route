"""Multi-Head Attention Layer for GAT Encoder.

Attributes:
    GATMultiHeadAttentionLayer: Single layer of the Graph Attention Encoder.

Example:
    >>> from logic.src.models.subnets.encoders.gat.gat_multi_head_attention_layer import GATMultiHeadAttentionLayer
    >>> layer = GATMultiHeadAttentionLayer(n_heads=8, embed_dim=128, feed_forward_hidden=512)
"""

from __future__ import annotations

from typing import Any, Optional

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import MultiHeadAttentionLayerBase


class GATMultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    """Single layer of the Graph Attention Encoder.

    Inherits from MultiHeadAttentionLayerBase, providing:
    - Multi-head attention with configurable connections (skip/dense/hyper)
    - Feed-forward network with configurable activation
    - Layer normalization
    - Automatic 4D tensor handling for hyper-connections

    Attributes:
        n_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        feed_forward_hidden (int): Hidden dimension for feed-forward network.
        norm_config (NormalizationConfig): Normalization configuration.
        activation_config (ActivationConfig): Activation function configuration.
        conn_type (str): Connection type.
        expansion_rate (int): Expansion factor for hyper-connections.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initializes the GATMultiHeadAttentionLayer.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for feed-forward network.
            norm_config: Normalization configuration.
            activation_config: Activation function configuration.
            connection_type: Connection type ("skip", "dense", or "hyper").
            expansion_rate: Expansion factor for hyper-connections.
            kwargs: Additional keyword arguments.
        """
        super(GATMultiHeadAttentionLayer, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
            **kwargs,
        )
