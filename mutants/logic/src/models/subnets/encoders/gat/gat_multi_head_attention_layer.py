"""Multi-Head Attention Layer for GAT Encoder."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from logic.src.models.subnets.modules.connections import get_connection_module

from .gat_feed_forward_sublayer import GATFeedForwardSubLayer


class GATMultiHeadAttentionLayer(nn.Module):
    """
    Single layer of the Graph Attention Encoder.
    Uses connections factory for potential hyper-connections.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        normalization: str,
        epsilon_alpha: float,
        learn_affine: bool,
        track_stats: bool,
        mbeta: float,
        lr_k: float,
        n_groups: int,
        activation: str,
        af_param: float,
        threshold: float,
        replacement_value: float,
        n_params: int,
        uniform_range: List[float],
        connection_type: str = "skip",
        expansion_rate: int = 4,
    ) -> None:
        """Initializes the GATMultiHeadAttentionLayer."""
        super(GATMultiHeadAttentionLayer, self).__init__()

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

        self.ff = get_connection_module(
            module=GATFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation,
                af_param,
                threshold,
                replacement_value,
                n_params,
                uniform_range,
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

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        """
        h = self.att(h, mask=mask)

        # Handle Norm for Hyper-Connections (4D input)
        if h.dim() == 4:  # (B, S, D, n)
            # Permute to (B, S, n, D) for Norm(D)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm: torch.Tensor = self.norm1(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            h = self.norm1(h)

        h = self.ff(h)

        if h.dim() == 4:
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm = self.norm2(h_perm)
            h = h_norm.permute(0, 1, 3, 2)
        else:
            res: torch.Tensor = self.norm2(h)
            return res

        return h
