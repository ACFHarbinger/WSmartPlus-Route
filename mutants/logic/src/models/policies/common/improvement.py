"""
Improvement Policy Base Classes.

This module provides base classes for improvement-based models.
Improvement models start with an initial solution and iteratively
improve it through local search or neural modification steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict


class ImprovementEncoder(nn.Module, ABC):
    """
    Base class for improvement encoders.

    Improvement encoders take both the problem instance and the current solution
    to produce embeddings representing the current state of search.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize ImprovementEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Compute embeddings for the problem instance and current solution.

        Args:
            td: TensorDict containing problem instance and 'tour' or current solution.

        Returns:
            Embeddings tensor or tuple of tensors.
        """
        raise NotImplementedError


class ImprovementDecoder(nn.Module, ABC):
    """
    Base class for improvement decoders.

    Improvement decoders predict moves or operators to apply to the current solution.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize ImprovementDecoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: RL4COEnvBase,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict improvement moves.

        Args:
            td: TensorDict containing current state.
            embeddings: Encoded state and solution.
            env: Environment for state transitions.

        Returns:
            Tuple of (log_likelihood, actions/moves).
        """
        raise NotImplementedError


class ImprovementPolicy(nn.Module, ABC):
    """
    Base class for improvement policies.

    Improvement policies take an instance + a solution as input and output a specific
    operator that changes the current solution to a new one.
    """

    def __init__(
        self,
        encoder: Optional[ImprovementEncoder] = None,
        decoder: Optional[ImprovementDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize ImprovementPolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state and current solution.
            env: Environment to use for decoding.
            phase: Phase of the algorithm (train, val, test).
            return_actions: Whether to return the actions.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally actions.
        """
        raise NotImplementedError
