from __future__ import annotations

import torch
import torch.nn as nn

from logic.src.models.modules.multi_head_attention import MultiHeadAttention
from logic.src.models.modules.normalization import Normalization


class MatNetEncoderLayer(nn.Module):
    """
    Encoder layer for MatNet.
    Performs row and column-wise attention on matrix embeddings.
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

        self.row_mha = MultiHeadAttention(n_heads, embed_dim, embed_dim)
        self.col_mha = MultiHeadAttention(n_heads, embed_dim, embed_dim)

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

    def forward(self, row_emb: torch.Tensor, col_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            row_emb: [batch, row_size, embed_dim]
            col_emb: [batch, col_size, embed_dim]
        """
        # Row-wise MHA
        row_res = row_emb
        row_emb = self.row_mha(row_emb)
        row_emb = self.row_norm1(row_emb + row_res)
        row_res = row_emb
        row_emb = self.row_ff(row_emb)
        row_emb = self.row_norm2(row_emb + row_res)

        # Col-wise MHA
        col_res = col_emb
        col_emb = self.col_mha(col_emb)
        col_emb = self.col_norm1(col_emb + col_res)
        col_res = col_emb
        col_emb = self.col_ff(col_emb)
        col_emb = self.col_norm2(col_emb + col_res)

        # Cross-interaction (simplified as self-attention + residual for now,
        # actual MatNet uses mixed score matrix which is more complex)
        # Note: In original MatNet, row and col embeddings interact via the distance matrix
        # and specialized attention heads. This is a baseline implementation.

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

    def forward(self, row_emb: torch.Tensor, col_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb)
        return row_emb, col_emb
