"""Mixture of Experts Temporal Attention Model.

This module provides `MoETemporalAttentionModel`, which combines the
time-varying graph processing of `TemporalAttentionModel` with the sparse
capacity of Mixture of Experts.

Attributes:
    MoETemporalAttentionModel: Temporal model with expert-routed encoding.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from logic.src.models.core.temporal_attention_model import TemporalAttentionModel
from logic.src.models.subnets.factories import MoEComponentFactory


class MoETemporalAttentionModel(TemporalAttentionModel):
    """Temporal Attention Model with sparse Mixture of Experts (MoE).

    Handles evolving problem states (e.g., dynamic waste collection) while
    leveraging expert sub-networks to specialize on different temporal dynamics or
    spatial regions.

    Attributes:
        total_experts (int): Global count of active experts.
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
        """Initializes the MoE Temporal Attention Model.

        Args:
            embed_dim: dimensionality of latent features.
            hidden_dim: multi-expert sub-layer width.
            problem: domain context.
            n_encode_layers: transformer depth.
            n_encode_sublayers: multi-context sub-layer count.
            n_decode_layers: decoder block count.
            dropout_rate: probability of feature zeroing.
            normalization: type of feature normalization.
            n_heads: attention heads.
            num_experts: population of experts per layer.
            k: active experts per token.
            noisy_gating: enable stochastic routing.
            **kwargs: extra parameters for base TemporalAttentionModel.
        """
        # Create the MoE Factory for component instantiation
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

    def embed_and_transform(
        self, input: Any, edges: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encodes temporal input and collapses high-dimensional expert outputs.

        Wraps the base transformation to handle potential 4D tensors produced by
        multi-expert routing, reducing them to standard graph embeddings.

        Args:
            input: raw graph features or sequence.
            edges: connectivity information.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - transformed embeddings [B, N, D].
                - auxiliary edge features (None).
        """
        embeddings, _ = super().embed_and_transform(input, edges)  # type: ignore[misc]

        if embeddings.dim() == 4:
            # Collapse the expert/head dimension: [Batch, N, K, Dim] -> [Batch, N, Dim]
            embeddings = embeddings.mean(dim=2)

        return embeddings, None

    @property
    def total_experts(self) -> int:
        """Retrieves count of experts in the model.

        Returns:
            int: expert module count.
        """
        return self.encoder.num_experts if hasattr(self.encoder, "num_experts") else 0
