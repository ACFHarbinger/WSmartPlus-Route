"""Connection factory for various neural network architectures.

This module provides the get_connection_module utility, which wraps existing
layers in residual (skip) or state-dependent hyper-connections.

Attributes:
    get_connection_module: Factory for instantiating connection wrappers.

Example:
    >>> from torch import nn
    >>> from logic.src.models.subnets.modules.connections import get_connection_module
    >>> module = nn.Linear(128, 128)
    >>> skip = get_connection_module(module, 128, connection_type="skip")
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .dynamic_hyper_connection import DynamicHyperConnection
from .skip_connection import SkipConnection
from .static_hyper_connection import StaticHyperConnection


class Connections(nn.Module):
    """Factory module for creating connection wrappers.

    Note: This class primarily group connection-related logic and usually
    instantiations happen through the functional `get_connection_module`.

    Attributes:
        None: This module serves as a pure factory logic container.
    """

    def __init__(self) -> None:
        """Initializes the connections factory module."""
        super().__init__()


def get_connection_module(
    module: nn.Module,
    embed_dim: int,
    connection_type: str = "skip",
    **kwargs: Any,
) -> nn.Module:
    """Returns a connection module for the given type.

    Wraps a base module in a residual or hyper-connection interface.

    Args:
        module: The sub-module to wrap (e.g., Attention, FeedForward).
        embed_dim: Sequence feature dimensionality.
        connection_type: Pattern to use. Options are 'skip' (residual),
            'static_hyper', or 'dynamic_hyper'. Defaults to 'skip'.
        kwargs: Additional arguments for the chosen connection module.

    Returns:
        nn.Module: The wrapped connection module.

    Raises:
        ValueError: If the connection type is not recognized.
    """
    if connection_type in {"skip", "residual"}:
        return SkipConnection(module)
    elif connection_type == "static_hyper":
        return StaticHyperConnection(module, embed_dim, **kwargs)
    elif connection_type == "dynamic_hyper":
        return DynamicHyperConnection(module, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")
