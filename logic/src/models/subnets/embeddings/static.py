"""Static embedding module.

This module provides the StaticEmbedding layer, which returns zero tensors for
all dynamic updates, effectively disabling dynamic attention keys and values.

Attributes:
    StaticEmbedding: Module that provides zero-updates for dynamic attention.

Example:
    >>> from logic.src.models.subnets.embeddings.static import StaticEmbedding
    >>> embed = StaticEmbedding()
    >>> d_k, d_v, d_l = embed(td)
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from tensordict import TensorDict
from torch import nn


class StaticEmbedding(nn.Module):
    """Static embedding: No dynamic updates during decoding.

    This class serves as a placeholder or baseline where no dynamic features
    (like visited masks) are used to update the attention mechanism components.

    Attributes:
        args: Captured positional arguments.
        kwargs: Captured keyword arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes StaticEmbedding.

        Args:
            args: Captured positional arguments.
            kwargs: Captured keyword arguments.
        """
        super().__init__()

    def forward(self, td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns zero tensors for all dynamic updates.

        Args:
            td: Current state dictionary.

        Returns:
            Tuple: Three zero-valued tensors for (K_glimpse, V_glimpse, K_logit).
        """
        return (
            torch.tensor(0.0, device=td.device),
            torch.tensor(0.0, device=td.device),
            torch.tensor(0.0, device=td.device),
        )
