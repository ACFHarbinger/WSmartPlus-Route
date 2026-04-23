"""MatNet Attention Module.

This module provides the MixedScoreMHA layer, which implements the dual-stream
attention mechanism for matching problems (e.g., assignment problems) as
proposed in MatNet.

Attributes:
    MixedScoreMHA: Dual-stream attention mechanism for matching problems.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.matnet_attention import MixedScoreMHA
    >>> row_emb = torch.randn(1, 4, 128)
    >>> col_emb = torch.randn(1, 4, 128)
    >>> matrix = torch.randn(1, 4, 4)
    >>> model = MixedScoreMHA(n_heads=8, embed_dim=128)
    >>> row_out, col_out = model(row_emb, col_emb, matrix)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MixedScoreMHA(nn.Module):
    """Mixed-Score Multi-Head Attention for MatNet.

    Computes attention scores by combining row and column representations with
    raw cost matrix elements, allowing for structured cross-representation communication.

    Reference:
        Kwon, Y. D., et al. (2021). MatNet: Matrix-based Neural Networks for
        Combinatorial Optimization. Advances in Neural Information Processing Systems.

    Attributes:
        n_heads (int): Number of parallel attention heads.
        embed_dim (int): Input and output feature dimensionality.
        key_dim (int): Dimensionality of keys and queries per head.
        norm_factor (float): Scaling factor for dot-product attention (1 / sqrt(key_dim)).
        W_query_row (nn.Linear): Dimensionality mapping for row queries.
        W_key_row (nn.Linear): Dimensionality mapping for row keys.
        W_val_row (nn.Linear): Dimensionality mapping for row values.
        W_query_col (nn.Linear): Dimensionality mapping for column queries.
        W_key_col (nn.Linear): Dimensionality mapping for column keys.
        W_val_col (nn.Linear): Dimensionality mapping for column values.
        W_mat (nn.Parameter): Learnable scaling factor for the cost matrix elements.
        W_out_row (nn.Linear): Final output projection for row update.
        W_out_col (nn.Linear): Final output projection for column update.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        key_dim: Optional[int] = None,
    ) -> None:
        """Initializes MixedScoreMHA.

        Args:
            n_heads: Number of parallel attention heads.
            embed_dim: Feature dimensionality of row and column embeddings.
            key_dim: Dimensionality of internal keys/queries. Defaults to `embed_dim // n_heads`.
        """
        super().__init__()
        if key_dim is None:
            key_dim = embed_dim // n_heads

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)

        # Projections for row embeddings
        self.W_query_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_key_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_val_row = nn.Linear(embed_dim, n_heads * key_dim, bias=False)

        # Projections for column embeddings
        self.W_query_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_key_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)
        self.W_val_col = nn.Linear(embed_dim, n_heads * key_dim, bias=False)

        # Weight for the distance matrix scores
        self.W_mat = nn.Parameter(torch.Tensor(n_heads, 1, 1))

        self.W_out_row = nn.Linear(n_heads * key_dim, embed_dim, bias=False)
        self.W_out_col = nn.Linear(n_heads * key_dim, embed_dim, bias=False)

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes weight parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W_mat)

    def forward(
        self,
        row_emb: torch.Tensor,
        col_emb: torch.Tensor,
        matrix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixed-score forward pass computing mutual attention between rows and columns.

        Args:
            row_emb: Row embeddings of shape (batch, row_size, dim).
            col_emb: Column embeddings of shape (batch, col_size, dim).
            matrix: External cost/similarity matrix of shape (batch, row_size, col_size).
            mask: Optional tensor mask for invalid entries.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated row embeddings of shape (batch, row_size, dim).
                - Updated column embeddings of shape (batch, col_size, dim).
        """
        batch_size, row_size, _ = row_emb.size()

        def project(linear: nn.Linear, emb: torch.Tensor) -> torch.Tensor:
            """Projects and reshapes embedding for multi-head attention.

            Args:
                linear: Projection layer.
                emb: Input embedding.

            Returns:
                torch.Tensor: Reshaped projection (batch, heads, size, key_dim).
            """
            out = linear(emb)
            return out.view(batch_size, -1, self.n_heads, self.key_dim).transpose(1, 2)

        row_q = project(self.W_query_row, row_emb)
        row_k = project(self.W_key_row, row_emb)
        row_v = project(self.W_val_row, row_emb)

        col_q = project(self.W_query_col, col_emb)
        col_k = project(self.W_key_col, col_emb)
        col_v = project(self.W_val_col, col_emb)

        # Mixed scores: (Row_Q @ Col_K^T) + (Col_Q @ Row_K^T)^T + Matrix_contribution
        score1 = torch.matmul(row_q, col_k.transpose(-2, -1))  # [batch, n_heads, row_size, col_size]
        score2 = torch.matmul(col_q, row_k.transpose(-2, -1)).transpose(-2, -1)  # [batch, n_heads, row_size, col_size]
        score3 = matrix.unsqueeze(1) * self.W_mat  # [batch, n_heads, row_size, col_size]

        compatibility = self.norm_factor * (score1 + score2 + score3)

        if mask is not None:
            if mask.dim() == 3:
                compatibility.masked_fill_(mask.unsqueeze(1), -1e9)
            else:
                compatibility.masked_fill_(mask, -1e9)

        # Attention row-wise (row attending to cols)
        attn_row = torch.softmax(compatibility, dim=-1)
        # Attention col-wise (col attending to rows)
        attn_col = torch.softmax(compatibility, dim=-2)

        # New row embeddings from col values
        row_heads = torch.matmul(attn_row, col_v)  # [batch, n_heads, row_size, key_dim]
        # New col embeddings from row values
        col_heads = torch.matmul(attn_col.transpose(-2, -1), row_v)  # [batch, n_heads, col_size, key_dim]

        def project_out(linear: nn.Linear, heads: torch.Tensor) -> torch.Tensor:
            """Reshapes and applies final linear transformation to heads.

            Args:
                linear: Output projection layer.
                heads: Multi-head attention outputs.

            Returns:
                torch.Tensor: Combined output tensor.
            """
            out = heads.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.key_dim)
            return linear(out)

        row_out = project_out(self.W_out_row, row_heads)
        col_out = project_out(self.W_out_col, col_heads)

        return row_out, col_out
