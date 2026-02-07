from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from logic.src.models.modules.multi_head_attention_matnet import MixedScoreMHA
from logic.src.models.modules.normalization import Normalization


class MatNetEncoderLayer(nn.Module):
    """
    Encoder layer for MatNet.
    Performs row and column-wise mixed-score attention on matrix embeddings.
    Reference: MatNet: Matrix Encoding Network for Combinatorial Optimization (Kwon et al., 2021)
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "instance",
    ):
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
        self, row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            row_emb: [batch, row_size, embed_dim]
            col_emb: [batch, col_size, embed_dim]
            matrix: [batch, row_size, col_size]
            mask: [batch, row_size, col_size]
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


class MatNetEncoder(nn.Module):
    """
    MatNet Encoder.
    Stacks multiple MatNetEncoderLayer.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        n_heads: int,
        feed_forward_hidden: int = 512,
        normalization: str = "instance",
    ):
        super(MatNetEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MatNetEncoderLayer(embed_dim, n_heads, feed_forward_hidden, normalization) for _ in range(num_layers)]
        )

    def forward(
        self, row_emb: torch.Tensor, col_emb: torch.Tensor, matrix: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, matrix, mask)
        return row_emb, col_emb
