"""Positional embedding modules for sequence-aware routing.

This package provides positional embedding layers that encode the relative
or absolute order of items, useful for sequence-based models like APE/CPE.

Attributes:
    POSITIONAL_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of positional
        embedding names to their respective classes.

Example:
    >>> from logic.src.models.subnets.embeddings.positional import pos_init_embedding
    >>> embedder = pos_init_embedding("APE", embed_dim=128)
"""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .absolute_positional_embedding import AbsolutePositionalEmbedding
from .cyclic_positional_embedding import CyclicPositionalEmbedding


def pos_init_embedding(
    pos_name: str,
    embed_dim: int,
    **kwargs: Any,
) -> nn.Module:
    """Factory function for positional embedding modules.

    Args:
        pos_name: Embedding type ("APE" for absolute, "CPE" for cyclic).
        embed_dim: Resulting embedding dimensionality.
        kwargs: Additional arguments passed to the specific constructor.

    Returns:
        nn.Module: Initialized positional embedding module.

    Raises:
        ValueError: If the positional embedding name is not recognized.
    """
    name_upper = pos_name.upper()
    if name_upper == "APE":
        return AbsolutePositionalEmbedding(embed_dim, **kwargs)
    elif name_upper == "CPE":
        return CyclicPositionalEmbedding(embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown positional embedding type: {pos_name}")


POSITIONAL_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "ape": AbsolutePositionalEmbedding,
    "cpe": CyclicPositionalEmbedding,
}

__all__: list[str] = [
    "AbsolutePositionalEmbedding",
    "CyclicPositionalEmbedding",
    "pos_init_embedding",
    "POSITIONAL_EMBEDDING_REGISTRY",
]
