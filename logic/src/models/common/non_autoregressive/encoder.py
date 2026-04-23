"""Non-autoregressive Encoder base module.

This module provides the abstract base class for encoders used in
non-autoregressive (NAR) models. These models predict a global heatmap
(e.g., edge probabilities) in a single pass over the problem instance.

Attributes:
    NonAutoregressiveEncoder: Abstract base class for NAR encoders.

Example:
    >>> from logic.src.models.common.non_autoregressive.encoder import NonAutoregressiveEncoder
    >>> # subclass and implement forward...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class NonAutoregressiveEncoder(nn.Module, ABC):
    """Base class for non-autoregressive encoders.

    NAR encoders take a problem instance and produce heatmaps
    (log probabilities over edges or nodes) in a single forward pass,
    instead of sequentially constructing embeddings.

    Attributes:
        embed_dim (int): Dimensionality of the produced heatmaps/embeddings.
    """

    def __init__(self, embed_dim: int = 128, **kwargs: Any) -> None:
        """Initializes the NonAutoregressiveEncoder.

        Args:
            embed_dim: Internal dimensionality for feature representations.
            **kwargs: Additional parameters passed to the parent Module.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Computes heatmap logits for the given problem instance.

        Args:
            td: TensorDict containing instance metadata (e.g., coordinates).
            **kwargs: Additional control arguments for heatmap generation.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
                Heatmap tensor of shape [batch, num_nodes, num_nodes] for
                edge-based models, or [batch, num_nodes] for node-based models.
        """
        raise NotImplementedError
