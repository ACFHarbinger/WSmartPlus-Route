"""Cyclic positional encoding module.

This module provides the CyclicPositionalEmbedding layer, which encodes relative
order using coordinates on a unit circle (Ma et al., 2021).

Attributes:
    CyclicPositionalEmbedding: Circular coordinate positional encoder.

Example:
    >>> from logic.src.models.subnets.embeddings.positional.cyclic_positional_embedding import CyclicPositionalEmbedding
    >>> pe = CyclicPositionalEmbedding(embed_dim=128)
    >>> x = pe(x, positions)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CyclicPositionalEmbedding(nn.Module):
    """Cyclic positional encoding (Ma et al. 2021).

    Projects normalized positions [0, 1] onto a unit circle (cosine and sine)
    and then maps them into the latent space to provide a translation-invariant
    order signal.

    Attributes:
        embed_dim (int): Dimensionality of the output embeddings.
        mean_pooling (bool): Whether to subtract the mean across the sequence length.
        linear (nn.Linear): Transformation layer from 2D cyclic coords to embed_dim.
    """

    def __init__(self, embed_dim: int, mean_pooling: bool = True) -> None:
        """Initializes CyclicPositionalEmbedding.

        Args:
            embed_dim: Internal embedding dimensionality.
            mean_pooling: If True, centers the generated encodings.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.mean_pooling = mean_pooling
        self.linear = nn.Linear(2, embed_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies cyclic positional encoding to the input sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            positions: Normalized sequence positions in [0, 1] (batch, seq_len).

        Returns:
            torch.Tensor: Input tensor with cyclic order information added.
        """
        # Convert to cyclic coordinates
        angles = 2 * math.pi * positions
        cyclic = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pe = self.linear(cyclic)

        if self.mean_pooling:
            pe = pe - pe.mean(dim=1, keepdim=True)

        return x + pe
