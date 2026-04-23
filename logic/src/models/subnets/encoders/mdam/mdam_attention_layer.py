"""Attention layer for MDAM Encoder.

Attributes:
    MultiHeadAttentionLayer: Single transformer layer with MHA + FFN for MDAM encoder.

Example:
    >>> from logic.src.models.subnets.encoders.mdam.mdam_attention_layer import MultiHeadAttentionLayer
    >>> layer = MultiHeadAttentionLayer(embed_dim=128, num_heads=8)
"""

from __future__ import annotations

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import MultiHeadAttentionLayerBase


class MultiHeadAttentionLayer(MultiHeadAttentionLayerBase):
    """Single transformer layer with MHA + FFN for MDAM encoder.

    Inherits from MultiHeadAttentionLayerBase, which provides the standard
    transformer encoder layer pattern. This is a simplified version that:
    - Uses ReLU activation (hardcoded)
    - Uses simple skip connections (no hyper-connections)
    - Uses configurable normalization type

    Attributes:
        n_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        feed_forward_hidden (int): Hidden dimension for feed-forward layers.
        norm_config (NormalizationConfig): Normalization configuration.
        activation_config (ActivationConfig): Activation function configuration.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "batch",
    ) -> None:
        """Initializes the MultiHeadAttentionLayer.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            feed_forward_hidden: Hidden dimension for FFN.
            normalization: Type of normalization ('batch', 'layer', 'instance').
        """
        # Create config objects for base class
        norm_config = NormalizationConfig(norm_type=normalization)
        activation_config = ActivationConfig(name="relu")  # MDAM uses ReLU

        # Delegate to base class with simplified configuration
        super().__init__(
            n_heads=num_heads,
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            connection_type="skip",  # MDAM uses simple residual connections
            expansion_rate=4,  # Not used with "skip", but required parameter
        )
