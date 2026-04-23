"""Mixture of Experts Attention Model.

This module provides `MoEAttentionModel`, which inherits from the base
`AttentionModel` but injects a `MoEComponentFactory` to instantiate
sparse-routed expert layers within the transformer architectures.

Attributes:
    MoEAttentionModel: Capacity-boosted attention model with expert routing.
"""

from __future__ import annotations

from typing import Any, Optional

from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.subnets.factories import MoEComponentFactory


class MoEAttentionModel(AttentionModel):
    """Attention Model with sparse Mixture of Experts (MoE).

    Increases model capacity while maintaining sparse activation patterns. This
    leads to better task-specific specialization in heterogeneous data
    distributions.

    Attributes:
        total_experts (int): Accessor for the active expert population count.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_encode_layers: int = 2,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        normalization: str = "batch",
        n_heads: int = 8,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the MoE Attention Model.

        Args:
            embed_dim: Dimensionality of latent features.
            hidden_dim: Expansion width for sub-networks.
            problem: domain context.
            n_encode_layers: Transformer depth.
            n_encode_sublayers: Multi-context sub-layer count.
            n_decode_layers: Decoder block count.
            dropout_rate: Probability of zeroing weights.
            normalization: Type of feature normalization.
            n_heads: Parallel attention heads.
            num_experts: Population size of experts per layer.
            k: Top-k gating activation factor.
            noisy_gating: Enable router exploration noise.
            **kwargs: Extra parameters for the base AttentionModel.
        """
        # Create the MoE Factory to build expert-enabled subnets
        component_factory = MoEComponentFactory(num_experts=num_experts, k=k, noisy_gating=noisy_gating)

        super().__init__(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            problem=problem,
            component_factory=component_factory,
            n_encode_layers=n_encode_layers,
            n_encode_sublayers=n_encode_sublayers,
            n_decode_layers=n_decode_layers,
            dropout_rate=dropout_rate,
            normalization=normalization,
            n_heads=n_heads,
            **kwargs,
        )

    @property
    def total_experts(self) -> int:
        """Retrieves the global count of experts active in the encoder stack.

        Returns:
            int: Number of expert modules.
        """
        return self.encoder.num_experts if hasattr(self.encoder, "num_experts") else 0
