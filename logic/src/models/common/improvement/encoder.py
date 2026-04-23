"""Improvement Encoder base module.

This module provides the abstract base class for encoders used in improvement-based
neural models. These encoders process both the problem instance and the current
candidate solution to generate latent search space embeddings.

Attributes:
    ImprovementEncoder: Abstract base class for improvement-based encoders.

Example:
    >>> from logic.src.models.common.improvement.encoder import ImprovementEncoder
    >>> # subclass and implement forward...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class ImprovementEncoder(nn.Module, ABC):
    """Base class for improvement encoders.

    Improvement encoders take both the problem instance and the current solution
    to produce embeddings representing the current state of search.

    Attributes:
        embed_dim (int): Dimensionality of the resulting embeddings.
    """

    def __init__(self, embed_dim: int = 128, **kwargs: Any) -> None:
        """Initializes the ImprovementEncoder.

        Args:
            embed_dim: Internal dimensionality for encoding features.
            **kwargs: Additional parameters for the parent Module.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Computes embeddings for the problem instance and current solution.

        Args:
            td: TensorDict containing the problem instance and current solution
                (typically in a 'tour' or 'actions' field).
            **kwargs: Additional control arguments for encoding.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
                Encodings representing the search state, either as a single
                tensor or a tuple of feature-specific tensors.
        """
        raise NotImplementedError
