"""
Positional embeddings for transformer models.

Implements:
- AbsolutePositionalEmbedding: Standard sinusoidal (Vaswani et al. 2017)
- CyclicPositionalEmbedding: Cyclic encoding (Ma et al. 2021)
"""

import math

import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]


class CyclicPositionalEmbedding(nn.Module):
    """Cyclic positional encoding (Ma et al. 2021)."""

    def __init__(self, embed_dim: int, mean_pooling: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.mean_pooling = mean_pooling
        self.linear = nn.Linear(2, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cyclic positional encoding.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            positions: Normalized positions in [0, 1] (batch, seq_len)

        Returns:
            Tensor with cyclic positional encoding
        """
        # Convert to cyclic coordinates
        angles = 2 * math.pi * positions
        cyclic = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pe = self.linear(cyclic)

        if self.mean_pooling:
            pe = pe - pe.mean(dim=1, keepdim=True)

        return x + pe


def pos_init_embedding(
    pos_name: str,
    embed_dim: int,
    **kwargs,
) -> nn.Module:
    """Factory for positional embeddings.

    Args:
        pos_name: "APE" for absolute, "CPE" for cyclic
        embed_dim: Embedding dimension
        **kwargs: Additional arguments

    Returns:
        Positional embedding module
    """
    if pos_name.upper() == "APE":
        return AbsolutePositionalEmbedding(embed_dim, **kwargs)
    elif pos_name.upper() == "CPE":
        return CyclicPositionalEmbedding(embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown positional embedding: {pos_name}")
