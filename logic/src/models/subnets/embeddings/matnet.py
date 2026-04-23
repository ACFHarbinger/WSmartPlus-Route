"""MatNet Embedding module.

This module implements the MatNet initial embedding, which encodes row and column
statistics from cost matrices into high-dimensional space.

Attributes:
    MatNetInitEmbedding: Encoder for matrix-based inputs (e.g., cross-matrix metrics).

Example:
    >>> from logic.src.models.subnets.embeddings.matnet import MatNetInitEmbedding
    >>> embed = MatNetInitEmbedding(embed_dim=128)
    >>> row_emb, col_emb = embed(cost_matrix)
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class MatNetInitEmbedding(nn.Module):
    """Initial embedding layer for MatNet.

    Projects row and column statistics (mean, min) of the input cost matrix
    to generate dual embeddings for bipartite graphs or matrix-based CO problems.

    Reference:
        Kwon et al. "MatNet: Matrix Encoding Network for Combinatorial
        Optimization" (NeurIPS 2021)

    Attributes:
        row_stats_proj (nn.Linear): Linear projection for row-level statistics.
        col_stats_proj (nn.Linear): Linear projection for column-level statistics.
    """

    def __init__(self, embed_dim: int, normalization: str = "instance") -> None:
        """Initializes MatNetInitEmbedding.

        Args:
            embed_dim: Target embedding dimensionality for rows and columns.
            normalization: Type of normalization to apply (default: "instance").
        """
        super().__init__()
        # We project: mean, min, (and potentially others like max or std)
        # Standard MatNet paper uses mean and min.
        self.row_stats_proj = nn.Linear(2, embed_dim)
        self.col_stats_proj = nn.Linear(2, embed_dim)

    def forward(self, matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes matrix statistics into row and column embeddings.

        Args:
            matrix: Input cost/distance matrix of shape (batch, row_size, col_size).

        Returns:
            Tuple: Row embeddings (batch, row_size, dim) and column
                embeddings (batch, col_size, dim).
        """
        # Row stats: mean and min
        row_mean = matrix.mean(dim=2, keepdim=True)
        row_min, _ = matrix.min(dim=2, keepdim=True)
        row_stats = torch.cat([row_mean, row_min], dim=2)  # [batch, row_size, 2]
        row_emb = self.row_stats_proj(row_stats)

        # Col stats: mean and min
        col_mean = matrix.mean(dim=1, keepdim=True).transpose(1, 2)
        col_min, _ = matrix.min(dim=1, keepdim=True)
        col_min = col_min.transpose(1, 2)
        col_stats = torch.cat([col_mean, col_min], dim=2)  # [batch, col_size, 2]
        col_emb = self.col_stats_proj(col_stats)

        return row_emb, col_emb
