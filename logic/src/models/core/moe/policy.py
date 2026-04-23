"""Mixture of Experts (MoE) Policy for routing.

This module provides `MoEPolicy`, an extension of `AttentionModelPolicy` that
replaces the standard Feed-Forward Networks in the transformer encoder with
Sparse Mixture of Experts layers to increase model capacity without proportional
computational cost.

Attributes:
    MoEPolicy: Routing model with expert-routed attention encoding.
"""

from __future__ import annotations

from typing import Any, cast

from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder


class MoEPolicy(AttentionModelPolicy):
    """Mixture of Experts Policy.

    Utilizes `MoEGraphAttentionEncoder` to route node features through a subset
    of specialized 'expert' networks during encoding, allowing the model to
    specialize on different graph structures or tasks.

    Attributes:
        encoder (MoEGraphAttentionEncoder): Sparse-routed transformer encoder.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the Mixture of Experts policy.

        Args:
            env_name: Optimization task identifier.
            embed_dim: dimensionality of latent features.
            hidden_dim: Multi-layer perceptron expansion size.
            n_encode_layers: Depth of the contextual encoder.
            n_heads: Count of attention heads.
            normalization: Type of normalization layer.
            num_experts: Total count of expert sub-networks in each layer.
            k: number of experts active per token.
            noisy_gating: whether to add noise to routing to balance load.
            **kwargs: Extra parameters for base policy and encoder.
        """
        # We invoke super() but we will overwrite the encoder.
        # This is slightly inefficient (creates standard encoder then throws it away)
        # but prevents duplicating all the other setup logic in AttentionModelPolicy.
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            normalization=normalization,
            **kwargs,
        )

        # Overwrite encoder with MoE version
        self.encoder = cast(
            MoEGraphAttentionEncoder,
            MoEGraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embed_dim,
                feed_forward_hidden=hidden_dim,
                n_layers=n_encode_layers,
                normalization=normalization,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
                **kwargs,
            ),
        )
