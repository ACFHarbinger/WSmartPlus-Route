"""Positional Embedding Factory."""

from __future__ import annotations

import torch.nn as nn

from .absolute_positional_embedding import AbsolutePositionalEmbedding
from .cyclic_positional_embedding import CyclicPositionalEmbedding


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
