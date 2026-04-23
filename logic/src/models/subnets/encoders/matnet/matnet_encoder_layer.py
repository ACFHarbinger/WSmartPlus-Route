"""Encoder layer for MatNet.

Attributes:
    MatNetEncoderLayer: Encoder layer for MatNet performing row and column-wise mixed-score attention.

Example:
    >>> from logic.src.models.subnets.encoders.matnet.matnet_encoder_layer import MatNetEncoderLayer
    >>> layer = MatNetEncoderLayer(embed_dim=128, n_heads=8)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from logic.src.models.subnets.modules.matnet_attention import MixedScoreMHA
from logic.src.models.subnets.modules.normalization import Normalization


class MatNetEncoderLayer(nn.Module):
    """Encoder layer for MatNet.

    Performs row and column-wise mixed-score attention on matrix embeddings.
    Reference: MatNet: Matrix Encoding Network for Combinatorial Optimization
    (Kwon et al., 2021).

    Attributes:
        mha (MixedScoreMHA): Mixed-score multi-head attention module.
        row_norm1 (Normalization): First normalization layer for rows.
        col_norm1 (Normalization): First normalization layer for columns.
        row_ff (nn.Sequential): Feed-forward sublayer for rows.
        col_ff (nn.Sequential): Feed-forward sublayer for columns.
        row_norm2 (Normalization): Second normalization layer for rows.
        col_norm2 (Normalization): Second normalization layer for columns.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "instance",
    ) -> None:
        """Initializes the MatNetEncoderLayer.

        Args:
            embed_dim: Embedding dimension.
            n_heads: Number of attention heads.
            feed_forward_hidden: Hidden dimension for feed-forward layers.
            normalization: Type of normalization ("instance", "layer", or "batch").
        """
        super(MatNetEncoderLayer, self).__init__()

        self.mha = MixedScoreMHA(n_heads, embed_dim)

        self.row_norm1 = Normalization(embed_dim, normalization)
        self.col_norm1 = Normalization(embed_dim, normalization)

        self.row_ff = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim),
        )
        self.col_ff = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim),
        )

        self.row_norm2 = Normalization(embed_dim, normalization)
        self.col_norm2 = Normalization(embed_dim, normalization)

    def forward(
        self,
        row_emb: torch.Tensor,
        col_emb: torch.Tensor,
        matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            row_emb: Row embeddings of shape (batch, row_size, embed_dim).
            col_emb: Column embeddings of shape (batch, col_size, embed_dim).
            matrix: Input cost/distance matrix of shape (batch, row_size, col_size).
            mask: Optional mask tensor of shape (batch, row_size, col_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated row and column embeddings.
        """
        # Mixed-Score MHA
        row_res = row_emb
        col_res = col_emb

        row_attn_out, col_attn_out = self.mha(row_emb, col_emb, matrix, mask)

        row_emb = self.row_norm1(row_attn_out + row_res)
        col_emb = self.col_norm1(col_attn_out + col_res)

        # Feed-forward
        row_res = row_emb
        col_res = col_emb

        row_emb = self.row_ff(row_emb)
        col_emb = self.col_ff(col_emb)

        row_emb = self.row_norm2(row_emb + row_res)
        col_emb = self.col_norm2(col_emb + col_res)

        return row_emb, col_emb
