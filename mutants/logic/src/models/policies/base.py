"""
Base policy classes for constructive and improvement methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from logic.src.utils.functions.decoding import get_decoding_strategy
from tensordict import TensorDict


class ConstructivePolicy(nn.Module, ABC):
    """
    Base class for constructive (autoregressive) policies.

    Constructive policies build solutions step-by-step by selecting
    one node at a time until the solution is complete.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize ConstructivePolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> dict:
        """
        Full forward pass: encode + decode until done.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions.
            decode_type: Decoding strategy ("sampling", "greedy", "beam_search").
            num_starts: Number of solution starts for multi-start methods.

        Returns:
            Dictionary containing:
                - reward: Final reward/cost
                - log_likelihood: Log probability of solution
                - tour: Sequence of visited nodes
        """
        raise NotImplementedError

    def _select_action(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        decode_type: str = "sampling",
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action based on logits and decode type using DecodingStrategy.

        Args:
            logits: Action logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]
            decode_type: Decoding strategy name
            **kwargs: Additional arguments for decoding strategy (temperature, etc.)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        # Get strategy (can be cached if needed)
        strategy = get_decoding_strategy(decode_type, **kwargs)

        # Step
        result = strategy.step(logits, mask)
        if len(result) == 3:
            action, log_prob, entropy = result
        else:
            action, log_prob = result[:2]  # type: ignore[misc]
            entropy = result[2] if len(result) > 2 else torch.tensor(0.0)
        return action, log_prob, entropy


class ImprovementPolicy(nn.Module, ABC):
    """
    Base class for improvement-based policies.

    Improvement policies start with an initial solution and
    iteratively improve it through local search operations.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        env_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize ImprovementPolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        **kwargs,
    ) -> dict:
        """Execute improvement iterations."""
        raise NotImplementedError
