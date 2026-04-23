"""PolyNet Attention Module.

This module provides the PolyNetAttention layer, which implements K-strategy
conditioning via binary vectors to enable learning of diverse solution
strategies within a single neural network.

Reference:
    Hottung, A., et al. (2024). PolyNet: Learning Diverse Solution Strategies
    for Neural Combinatorial Optimization. arXiv preprint arXiv:2402.14048.

Attributes:
    PolyNetAttention: Attention mechanism with binary strategy conditioning.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.modules.polynet_attention import PolyNetAttention
    >>> poly = PolyNetAttention(k=4, embed_dim=128)
    >>> q = torch.randn(1, 1, 128)
    >>> k = torch.randn(1, 10, 128)
    >>> v = torch.randn(1, 10, 128)
    >>> lk = torch.randn(1, 10, 128)
    >>> logits = poly(q, k, v, lk)
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class PolyNetAttention(nn.Module):
    """PolyNet Attention with K-strategy binary conditioning.

    Extends standard multi-head attention to condition glimpse vectors on K different
    binary identifier vectors. This forces the model to branch its decision logic,
    facilitating the discovery of multiple distinct high-quality paths for complex
    optimization problems.

    Attributes:
        k (int): Number of distinct strategies to learn.
        embed_dim (int): Dimensionality of input features.
        num_heads (int): Number of attention heads.
        mask_inner (bool): Whether to mask padding in internal attention.
        check_nan (bool): Whether to perform runtime NaN checks on logits.
        binary_vector_dim (int): Number of bits required to encode k strategies.
        binary_vectors (torch.Tensor): Precomputed binary identifier vectors.
        project_out (nn.Linear): Linear output projection for concatenated heads.
        poly_layer_1 (nn.Linear): First conditioning transformation.
        poly_layer_2 (nn.Linear): Second conditioning transformation for residuals.
    """

    binary_vectors: torch.Tensor

    def __init__(
        self,
        k: int,
        embed_dim: int,
        poly_layer_dim: int = 256,
        num_heads: int = 8,
        mask_inner: bool = True,
        out_bias: bool = False,
        check_nan: bool = True,
    ) -> None:
        """Initializes PolyNetAttention.

        Args:
            k: Number of parallel strategies to learn.
            embed_dim: Dimensionality of inputs and internal representations.
            poly_layer_dim: Dimensionality of conditioning mlp hidden layer.
            num_heads: Number of parallel attention heads.
            mask_inner: Whether to block padding during internal attention pass.
            out_bias: Whether to use a bias term in the glimpse projection.
            check_nan: Whether to enable runtime sanity checks for invalid values.
        """
        super().__init__()

        self.k = k
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_inner = mask_inner
        self.check_nan = check_nan

        # Binary vector dimension: ceil(log2(k)) bits needed
        self.binary_vector_dim = max(1, math.ceil(math.log2(k)))

        # Generate K binary vectors as non-trainable buffer
        all_binary = list(itertools.product([0, 1], repeat=self.binary_vector_dim))
        binary_vectors = torch.tensor(all_binary[:k], dtype=torch.float32)
        self.register_buffer("binary_vectors", binary_vectors)

        # Output projection
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)

        # PolyNet layers: condition on strategy via binary vector
        self.poly_layer_1 = nn.Linear(embed_dim + self.binary_vector_dim, poly_layer_dim)
        self.poly_layer_2 = nn.Linear(poly_layer_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        logit_key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes K-conditioned attention logits.

        Args:
            query: Query feature tensor of shape (batch, n_steps, dim).
            key: Key feature tensor of shape (batch, nodes, dim).
            value: Value feature tensor of shape (batch, nodes, dim).
            logit_key: Specialized key for final dot product (batch, nodes, dim).
            attn_mask: Boolean attention mask (batch, nodes). True indicates valid.

        Returns:
            torch.Tensor: Computed logits of shape (batch, n_steps, nodes).
        """
        # Compute inner multi-head attention
        heads = self._inner_mha(query, key, value, attn_mask)

        # Project to glimpse
        glimpse = self.project_out(heads)

        # Apply K-strategy conditioning
        num_solutions = glimpse.shape[1]
        z = self._get_strategy_vectors(num_solutions, glimpse.device)
        z = z[None].expand(glimpse.shape[0], num_solutions, self.binary_vector_dim)

        # PolyNet conditioning layers
        poly_input = torch.cat([glimpse, z], dim=-1)
        poly_out = F.relu(self.poly_layer_1(poly_input))
        poly_out = self.poly_layer_2(poly_out)

        # Residual connection
        glimpse = glimpse + poly_out

        # Compute final logits via dot product with logit key
        logits = torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "PolyNet logits contain NaNs"

        return logits

    def _get_strategy_vectors(
        self,
        num_solutions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Retrieves and cycles precomputed binary strategy vectors.

        Args:
            num_solutions: Number of parallel paths being evaluated.
            device: Target device for the tensor returned.

        Returns:
            torch.Tensor: Binary strategy matrix.
        """
        # Repeat binary vectors to cover all solutions
        repeats = math.ceil(num_solutions / self.k)
        vectors = self.binary_vectors.repeat(repeats, 1)[:num_solutions]
        return vectors.to(device)

    def _inner_mha(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Internal multi-head attention implementation.

        Args:
            query: Input queries.
            key: Input keys.
            value: Input values.
            attn_mask: Padding/Adjacency mask.

        Returns:
            torch.Tensor: Aggregated multi-head representation.
        """
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)

        # Prepare mask for multi-head attention
        if self.mask_inner and attn_mask is not None:
            # Add dimensions for heads and query positions
            if attn_mask.ndim == 2:
                m = attn_mask.unsqueeze(1).unsqueeze(2)
            elif attn_mask.ndim == 3:
                m = attn_mask.unsqueeze(1)
            else:
                m = attn_mask
        else:
            m = None

        # Scaled dot-product attention
        scale = math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if m is not None:
            scores = scores.masked_fill(~m, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        heads = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq, dim) -> (batch, seq, heads * dim)
        batch_size = heads.size(0)
        seq_len = heads.size(2)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return heads

    def _make_heads(self, v: torch.Tensor) -> torch.Tensor:
        """Reshapes and transposes a tensor for multi-head attention.

        Args:
            v: Input tensor (batch, seq, dim).

        Returns:
            torch.Tensor: Reshaped tensor (batch, heads, seq, head_dim).
        """
        batch_size = v.size(0)
        seq_len = v.size(1)
        head_dim = self.embed_dim // self.num_heads
        return v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
