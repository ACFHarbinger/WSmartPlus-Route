"""Autoregressive Encoder base module.

This module provides the abstract base class for encoders used in
autoregressive models, transforming problem instances into initial latent
representations.

Attributes:
    AutoregressiveEncoder: Base class for autoregressive encoders.

Example:
    >>> encoder = AutoregressiveEncoder()
    >>> embeddings = encoder(td)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn


class AutoregressiveEncoder(nn.Module, ABC):
    """Base class for autoregressive encoders.

    AR encoders take a problem instance and produce initial embeddings
    that represent the static and dynamic features of the problem state.

    Attributes:
        embed_dim: Dimensionality of the produced embeddings.
    """

    def __init__(self, embed_dim: int = 128, **kwargs: Any) -> None:
        """Initialize the AutoregressiveEncoder.

        Args:
            embed_dim: Internal dimensionality for the latent embeddings.
            kwargs: Additional keyword arguments passed to the parent nn.Module.
        """
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Compute initial embeddings for the problem instance.

        Args:
            td: TensorDict containing the current problem/environment state.
            kwargs: Control parameters for the encoder forward pass.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
                Encodings representing the problem state, either as a single
                tensor or a tuple of feature tensors.

        """
        raise NotImplementedError
