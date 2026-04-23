"""Multi-Head Attention module for transformer-based models.

This module provides the MultiHeadAttention layer, which allows the model to
jointly attend to information from different representation subspaces.

Attributes:
    MultiHeadAttention: Implementation of scaled dot-product multi-head attention.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.multi_head_attention import MultiHeadAttention
    >>> mha = MultiHeadAttention(n_heads=8, input_dim=128, embed_dim=128)
    >>> q = torch.randn(1, 10, 128)
    >>> out = mha(q)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    Allows the model to jointly attend to information from different representation
    subspaces at different positions using scaled dot-product attention.

    Attributes:
        n_heads (int): Number of attention heads.
        input_dim (int): Dimensionality of the input features.
        embed_dim (int): Total embedding dimensionality (output dimension).
        val_dim (int): Dimensionality of value vectors per head.
        key_dim (int): Dimensionality of key/query vectors per head.
        norm_factor (float): Scaling factor for dot-product attention (1 / sqrt(key_dim)).
        W_query (nn.Parameter): Learnable query projection weights.
        W_key (nn.Parameter): Learnable key projection weights.
        W_val (nn.Parameter): Learnable value projection weights.
        W_out (nn.Parameter): Learnable output projection weights.
        last_attn (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): Cached
            last attention weights and mask.
    """

    last_attn: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

    def __init__(
        self,
        n_heads: int,
        input_dim: int,
        embed_dim: int,
        val_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
    ) -> None:
        """Initializes MultiHeadAttention.

        Args:
            n_heads: Number of parallel attention heads.
            input_dim: Feature dimensionality of the input sequence.
            embed_dim: Total dimensionality of the output representation.
            val_dim: Dimension of value vectors per head. Defaults to `embed_dim // n_heads`.
            key_dim: Dimension of key/query vectors per head. Defaults to `val_dim`.
        """
        super().__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor: float = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
        self.init_parameters()
        self.last_attn = (None, None)

    def init_parameters(self) -> None:
        """Initializes the attention weights using Xavier-like uniform distribution."""
        for param in self.parameters():
            if param.dim() > 0:
                stdv: float = 1.0 / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        q: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the multi-head attention update.

        Args:
            q: Queries tensor of shape (batch, n_query, input_dim).
            h: Key/Value keys tensor of shape (batch, graph_size, input_dim).
                Defaults to `q` (self-attention) if None.
            mask: Boolean attention mask (batch, n_query, graph_size).

        Returns:
            torch.Tensor: Context vectors of shape (batch, n_query, embed_dim).
        """
        if h is None:
            h = q

        batch_size, graph_size, input_dim = h.size()
        n_query: int = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat: torch.Tensor = h.contiguous().view(-1, input_dim)
        qflat: torch.Tensor = q.contiguous().view(-1, input_dim)

        shp: Tuple[int, int, int, int] = (self.n_heads, batch_size, graph_size, -1)
        shp_q: Tuple[int, int, int, int] = (self.n_heads, batch_size, n_query, -1)

        Q: torch.Tensor = torch.matmul(qflat, self.W_query).view(shp_q)
        K: torch.Tensor = torch.matmul(hflat, self.W_key).view(shp)
        V: torch.Tensor = torch.matmul(hflat, self.W_val).view(shp)

        compatibility: torch.Tensor = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(1, batch_size, n_query, graph_size)

            mask_expanded: torch.Tensor = mask.expand_as(compatibility)
            compatibility[mask_expanded] = -math.inf

        attn: torch.Tensor = torch.softmax(compatibility, dim=-1)

        if mask is not None:
            mask_expanded = mask.expand_as(compatibility)
            self.last_attn = (attn.detach().clone(), mask_expanded.detach().clone())
            attnc: torch.Tensor = attn.clone()
            attnc[mask_expanded] = 0
            attn = attnc
        else:
            self.last_attn = (attn.detach().clone(), None)

        heads: torch.Tensor = torch.matmul(attn[:, :, :, :graph_size], V)
        out: torch.Tensor = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out
