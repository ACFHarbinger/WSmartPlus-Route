"""Pointer Attention Mixture of Experts Module.

This module provides the PointerAttnMoE layer, which combines multi-head
attention for glimpse generation with a Mixture-of-Experts (MoE) projection
for final pointing. This architecture allows the model to select specialized
pointing strategies based on the current context.

Attributes:
    PointerAttnMoE: Pointer attention enhanced with MoE output projection.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.pointer_attn_moe import PointerAttnMoE
    >>> model = PointerAttnMoE(embed_dim=128, num_heads=8)
    >>> q = torch.randn(1, 1, 128)
    >>> k = torch.randn(1, 10, 128)
    >>> v = torch.randn(1, 10, 128)
    >>> lk = torch.randn(1, 10, 128)
    >>> logits = model(q, k, v, lk)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from logic.src.models.subnets.modules.moe_layer import MoE
from logic.src.models.subnets.modules.multi_head_attention import MultiHeadAttention


class PointerAttnMoE(nn.Module):
    """Pointer Attention with Mixture-of-Experts (MoE) output projection.

    Uses standard Multi-Head Attention to compute a context vector (glimpse), which
    is then passed through an MoE layer before the final dot-product pointing operation.
    This enables per-step specialization of the pointing mechanism.

    Attributes:
        embed_dim (int): Dimensionality of input features.
        check_nan (bool): Whether to check for NaN values in output logits.
        mask_inner (bool): Whether to strictly mask internal attention.
        mha (MultiHeadAttention): Internal module for generating the glimpse vector.
        moe_out (MoE): Specialized transformation layer for context-aware pointing.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        mask_inner: bool = True,
        check_nan: bool = True,
    ) -> None:
        """Initializes PointerAttnMoE.

        Args:
            embed_dim: Dimensionality of inputs and internal representations.
            num_heads: Number of parallel attention heads in the glimpse mechanism.
            num_experts: Number of parallel transformation experts available.
            k: Number of experts to activate per time step.
            noisy_gating: Whether to use stochastic noise in the gating mechanism.
            mask_inner: Whether to block padding during the glimpse attention pass.
            check_nan: Whether to perform runtime validity checks on logits.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.check_nan = check_nan
        self.mask_inner = mask_inner

        # Inner Multi-Head Attention to compute glimpse
        self.mha = MultiHeadAttention(n_heads=num_heads, input_dim=embed_dim, embed_dim=embed_dim)

        # MoE for output projection of the glimpse
        self.moe_out = MoE(
            input_size=embed_dim,
            output_size=embed_dim,
            num_experts=num_experts,
            k=k,
            noisy_gating=noisy_gating,
            out_bias=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        logit_key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes pointer logits through MHA context and MoE projection.

        Args:
            query: Query feature tensor of shape (batch, n_steps, dim).
            key: Key database tensor of shape (batch, nodes, dim).
            value: Value database tensor of shape (batch, nodes, dim).
            logit_key: Pointing key database of shape (batch, nodes, dim).
            attn_mask: Boolean attention mask (batch, nodes). True indicates valid.

        Returns:
            torch.Tensor: Pointing logits of shape (batch, n_steps, nodes).
        """
        batch_size, num_steps, _ = query.shape

        # 1. Compute glimpse using MHA
        mha_mask = attn_mask
        if self.mask_inner and attn_mask is not None and attn_mask.dim() == 2:
            # (batch, nodes) -> (batch, num_steps, nodes)
            mha_mask = attn_mask.unsqueeze(1).expand(-1, num_steps, -1)

        glimpse = self.mha(q=query, h=key, mask=mha_mask)

        # 2. Project glimpse using MoE
        glimpse_moe = self.moe_out(glimpse)

        # 3. Compute logits via dot product
        # (B, N_steps, E) x (B, E, N_nodes) -> (B, N_steps, N_nodes)
        logits = torch.matmul(glimpse_moe, logit_key.transpose(-2, -1))

        # Scale
        logits = logits / math.sqrt(self.embed_dim)

        if self.check_nan:
            assert not torch.isnan(logits).any(), "PointerAttnMoE logits contain NaNs"

        return logits
