"""
Multi-Head Attention module for transformer-based models.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different representation subspaces
    at different positions. Implements scaled dot-product attention (SDPA).
    now supports Flash Attention via PyTorch's SDPA.
    """

    last_attn: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
        bias: bool = True,
        attention_dropout: float = 0.0,
        store_attn_weights: bool = False,
    ) -> None:
        """
        Args:
            n_heads: Number of attention heads.
            input_dim: Dimensionality of the input features.
            embed_dim: Total embedding dimension (output dimension).
            val_dim: Deprecated/Ignored. Inferred from embed_dim.
            key_dim: Deprecated/Ignored. Inferred from embed_dim.
            bias: Whether to use bias in linear projections.
            attention_dropout: Dropout rate for attention weights.
            store_attn_weights: If True, uses manual attention to store weights.
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.store_attn_weights = store_attn_weights
        self.attention_dropout = attention_dropout

        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"

        # Use nn.Linear for projections (rl4co pattern)
        # Note: We use a single Wqkv for self-attention optimization if we were fully aligning,
        # but to support standard forward(q, h) where h can be different, we usually want
        # separate (q) and (k, v) projections if we want to be safe, OR we can stick to
        # separate Wq, Wk, Wv like before but using nn.Linear.
        # The previous implementation had manual Parameter tensors.
        # Using separate Linears is safer for the generic q != h case.

        self.W_query = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_key = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_val = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.init_parameters()
        self.last_attn = (None, None)

        # SDPA availability
        try:
            from torch.nn.functional import scaled_dot_product_attention

            self.sdpa_fn = scaled_dot_product_attention
        except ImportError:
            self.sdpa_fn = None
            if not store_attn_weights:
                # Warning: SDPA not found, falling back to manual
                pass

    def init_parameters(self) -> None:
        """Initializes the parameters of the attention layers using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the multi-head attention.

        Args:
            q: Queries tensor of shape (batch_size, n_query, input_dim).
            h: Key/Value keys tensor of shape (batch_size, graph_size, input_dim).
               If None, defaults to q (self-attention).
            mask: Attention mask of shape (batch_size, n_query, graph_size).
                  True = masked out (invalid).

        Returns:
            context vectors (aggregated values) of shape (batch_size, n_query, embed_dim).
        """
        if h is None:
            h = q

        batch_size = q.size(0)
        n_query = q.size(1)
        graph_size = h.size(1)

        # Project and reshape
        # [batch, n, input_dim] -> [batch, n, heads, head_dim] -> [batch, heads, n, head_dim]
        # We use reshape logic compatible with SDPA

        Q = self.W_query(q).view(batch_size, n_query, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_val(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)

        # Handle mask
        input_mask = None
        if mask is not None:
            # Expand mask for attention
            if mask.dim() == 2:
                # [batch, graph_size] -> [batch, 1, 1, graph_size]
                input_mask = mask.view(batch_size, 1, 1, graph_size)
            elif mask.dim() == 3:
                # [batch, n_query, graph_size] -> [batch, 1, n_query, graph_size]
                input_mask = mask.unsqueeze(1)
            else:
                input_mask = mask

        if self.store_attn_weights or self.sdpa_fn is None:
            return self._manual_attention(Q, K, V, input_mask)
        else:
            # Invert mask for SDPA (True = valid/keep)
            # WSmart: True = masked (invalid)
            sdpa_mask = ~input_mask if input_mask is not None else None

            # SDPA: [batch, heads, query_len, head_dim]
            out = self.sdpa_fn(Q, K, V, attn_mask=sdpa_mask, dropout_p=self.attention_dropout if self.training else 0.0)

            # Reshape back: [batch, heads, n_query, head_dim] -> [batch, n_query, embed_dim]
            out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
            return self.W_out(out)

    def _manual_attention(self, Q, K, V, mask):
        # Q, K, V are [batch, heads, len, dim]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        # Store for visualization
        # Legacy compatibility: last_attn = (attn, mask)
        # Note: legacy stored detached clones.
        self.last_attn = (attn_weights.detach(), mask.detach() if mask is not None else None)

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(Q.size(0), Q.size(2), self.embed_dim)
        return self.W_out(out)
