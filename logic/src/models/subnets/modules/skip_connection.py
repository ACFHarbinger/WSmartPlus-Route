"""Residual skip connection implementation.

This module provides the SkipConnection wrapper, which implements the
identity-based residual mapping (input + transformation).

Attributes:
    SkipConnection: Class for adding residual connections to neural modules.

Example:
    >>> from torch import nn
    >>> from logic.src.models.subnets.modules.skip_connection import SkipConnection
    >>> linear = nn.Linear(128, 128)
    >>> residual_linear = SkipConnection(linear)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SkipConnection(nn.Module):
    """Implements a residual connection: output = input + module(input).

    This wrapper ensures that the gradient can flow directly through the identity
    matching part of the network, helping to mitigate the vanishing gradient problem.

    Attributes:
        module (nn.Module): The transformation module being bridged by the skip.
    """

    def __init__(self, module: nn.Module) -> None:
        """Initializes SkipConnection.

        Args:
            module: The neural network module to wrap with a residual connection.
        """
        super().__init__()
        self.module = module

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Applies the residual skip connection: input + module(input).

        Args:
            input: Input feature tensor.
            args: Positional arguments for the wrapped `module`.
            kwargs: Keyword arguments for the wrapped `module`.

        Returns:
            torch.Tensor: The sum of the original input and the module's output.
        """
        return input + self.module(input, *args, **kwargs)
