"""MoE Multi-Head Attention Layer."""

from __future__ import annotations

import torch.nn as nn

from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from logic.src.models.subnets.modules.connections import get_connection_module
from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward


class MoEMultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the MoE Graph Attention Encoder.
    """

    def __init__(
        self,
        n_heads,
        embed_dim,
        feed_forward_hidden,
        normalization,
        epsilon_alpha,
        learn_affine,
        track_stats,
        mbeta,
        lr_k,
        n_groups,
        activation,
        af_param,
        threshold,
        replacement_value,
        n_params,
        uniform_range,
        connection_type="skip",
        expansion_rate=4,
        num_experts=4,
        k=2,
        noisy_gating=True,
    ):
        """Initializes the MoEMultiHeadAttentionLayer."""
        super(MoEMultiHeadAttentionLayer, self).__init__()

        self.att = get_connection_module(
            module=MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        self.norm1 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

        # Use MoEFeedForward instead of FeedForwardSubLayer
        self.ff = get_connection_module(
            module=MoEFeedForward(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        self.norm2 = Normalization(
            embed_dim,
            normalization,
            epsilon_alpha,
            learn_affine,
            track_stats,
            mbeta,
            n_groups,
            lr_k,
        )

    def forward(self, h, mask=None):
        """
        Forward pass.
        Args:
            h: Input features.
            mask: Attention mask.
        Returns:
            Updated features.
        """
        h = self.att(h, mask=mask)

        # Handle Norm for Hyper-Connections (4D input)
        if h.dim() == 4:  # (B, S, D, n)
            # Permute to (B, S, n, D) for Norm(D)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm1(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            h = self.norm1(h)

        h = self.ff(h)

        if h.dim() == 4:
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm2(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            return self.norm2(h)

        return h
