"""Sinusoidal positional embedding module.

This module provides the AbsolutePositionalEmbedding layer, which implements
standard sinusoidal positional encodings for transformer-based architectures.

Attributes:
    AbsolutePositionalEmbedding: Sine/Cosine absolute positional encoder.

Example:
    >>> from logic.src.models.subnets.embeddings.positional.absolute_positional_embedding import AbsolutePositionalEmbedding
    >>> pe = AbsolutePositionalEmbedding(embed_dim=128)
    >>> x = pe(x)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class AbsolutePositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding.

    Implements absolute positional encodings using alternating sine and cosine
    functions of different frequencies to represent item positions in a sequence.

    Attributes:
        embed_dim (int): Dimensionality of the embeddings.
        pe (torch.Tensor): Precomputed positional encoding buffer.
    """

    pe: torch.Tensor

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        """Initializes AbsolutePositionalEmbedding.

        Args:
            embed_dim: Internal embedding dimensionality.
            max_len: Maximum supported sequence length for the lookup buffer.
        """
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            torch.Tensor: Augmented tensor with sinusoidal features.
        """
        return x + self.pe[:, : x.size(1)]
