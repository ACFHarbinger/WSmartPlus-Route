"""Multi-Head Flash Attention module.

This module provides the MultiHeadFlashAttention layer, which leverages PyTorch's
hardware-optimized `scaled_dot_product_attention` (SDPA) for high-performance
computation on supported hardware (e.g., NVIDIA Ampere GPUs).

Attributes:
    MultiHeadFlashAttention: High-performance multi-head attention using SDPA.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.flash_attention import MultiHeadFlashAttention
    >>> mha = MultiHeadFlashAttention(n_heads=8, input_dim=128, embed_dim=128)
    >>> q = torch.randn(1, 10, 128)
    >>> out = mha(q)
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from torch import nn

try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    scaled_dot_product_attention = None  # type: ignore[assignment]


class MultiHeadFlashAttention(nn.Module):
    """Multi-Head Attention using PyTorch's SDPA (Flash Attention).

    Efficient implementation of multi-head attention leveraging optimized kernels.
    Includes a manual fallback paths for debugging or when explicit weight
    inspection is required.

    Attributes:
        n_heads (int): Number of attention heads.
        input_dim (int): Input feature dimensionality.
        embed_dim (int): Output embedding dimensionality.
        head_dim (int): Dimensionality of each attention head.
        store_attn_weights (bool): Whether to explicitly calculate and store weights.
        attention_dropout (float): Dropout probability for attention weights.
        W_query (nn.Linear): Linear transformation for query mapping.
        W_key (nn.Linear): Linear transformation for key mapping.
        W_val (nn.Linear): Linear transformation for value mapping.
        W_out (nn.Linear): Linear transformation for output consolidation.
        last_attn (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): Cached
            last attention weights and mask.
        sdpa_fn (Optional[Callable]): Reference to the SDPA kernel function.
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
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        """Initializes MultiHeadFlashAttention.

        Args:
            n_heads: Number of parallel attention heads.
            input_dim: Feature dimensionality of the input sequence.
            embed_dim: Total dimensionality of the output representation.
            val_dim: Custom dimension for values (for compatibility).
            key_dim: Custom dimension for keys (for compatibility).
            bias: Whether to add a bias term in the projection layers.
            attention_dropout: Dropout rate applied to attention weights.
            store_attn_weights: Whether to use the slow manual path to store weights.
            sdpa_fn: Custom replacement for the SDPA function.
        """
        super().__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.store_attn_weights = store_attn_weights
        self.attention_dropout = attention_dropout

        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by n_heads"

        self.W_query = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_key = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_val = nn.Linear(input_dim, embed_dim, bias=bias)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.init_parameters()
        self.last_attn = (None, None)

        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

    def init_parameters(self) -> None:
        """Initializes the linear layer weights using Xavier uniform distribution."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes multi-head attention using hardware-optimized memory paths.

        Args:
            q: Queries tensor of shape (batch, n_query, input_dim).
            h: Key/Value keys tensor of shape (batch, graph_size, input_dim).
                Defaults to `q` (self-attention) if None.
            mask: Boolean attention mask (batch, n_query, graph_size).
                True indicates a masked position.

        Returns:
            torch.Tensor: Attended context vectors of shape (batch, n_query, embed_dim).
        """
        if h is None:
            h = q

        batch_size = q.size(0)
        n_query = q.size(1)
        graph_size = h.size(1)

        # q: [batch, len, dim] -> [batch, len, heads, head_dim] -> [batch, heads, len, head_dim]
        Q = self.W_query(q).view(batch_size, n_query, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_val(h).view(batch_size, graph_size, self.n_heads, self.head_dim).transpose(1, 2)

        input_mask = None
        if mask is not None:
            if mask.dim() == 2:
                input_mask = mask.view(batch_size, 1, 1, graph_size)
            elif mask.dim() == 3:
                input_mask = mask.unsqueeze(1)
            else:
                input_mask = mask

        if self.store_attn_weights or self.sdpa_fn is None:
            return self._manual_attention(Q, K, V, input_mask)
        else:
            sdpa_mask = ~input_mask if input_mask is not None else None

            out = self.sdpa_fn(
                Q,
                K,
                V,
                attn_mask=sdpa_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )

            out = out.transpose(1, 2).contiguous().view(batch_size, n_query, self.embed_dim)
            return self.W_out(out)

    def _manual_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback manual attention implementation for weight inspection.

        Args:
            Q: Query projections.
            K: Key projections.
            V: Value projections.
            mask: Optional tensor mask.

        Returns:
            torch.Tensor: Computed attention output.
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        self.last_attn = (
            attn_weights.detach(),
            mask.detach() if mask is not None else None,
        )

        if self.attention_dropout > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(Q.size(0), Q.size(2), self.embed_dim)
        return self.W_out(out)
