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
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase


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

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        decode_type: str = "greedy",
        num_starts: int = 1,
        max_steps: Optional[int] = None,
        phase: str = "train",
        return_actions: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass of the policy using an iterative improvement loop.

        Args:
            td: TensorDict containing the environment state and current solution.
            env: Environment to use for decoding.
            decode_type: Decoding strategy (greedy, sampling, etc.).
            num_starts: Number of solution starts.
            max_steps: Maximum number of improvement steps.
            phase: Phase of the algorithm (train, val, test).
            return_actions: Whether to return the actions.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally actions.
        """
        if env is None:
            # Try to get from name or default to base
            from logic.src.envs import get_env

            env = get_env(self.env_name or "tsp_kopt")

        # Initial solution generation (done by env.reset)
        td = env.reset(td)

        # Batch for multiple starts if requested
        from logic.src.utils.functions.decoding import batchify, unbatchify

        if num_starts > 1:
            td = batchify(td, num_starts)

        # Iterative improvement loop
        log_probs = []
        actions = []

        # Default steps from td or config
        if max_steps is None:
            max_steps = td.get("max_steps", torch.tensor(10)).item()

        for i in range(max_steps):
            # 1. Encode current state
            if self.encoder is None:
                raise ValueError("Encoder must be provided for ImprovementPolicy")
            embeddings = self.encoder(td)

            # 2. Predict move via decoder
            if self.decoder is None:
                raise ValueError("Decoder must be provided for ImprovementPolicy")
            log_p, move = self.decoder(td, embeddings, env, decode_type=decode_type, **kwargs)

            # 3. Apply move
            td.set("action", move)
            td = env.step(td)["next"]

            log_probs.append(log_p)
            actions.append(move)

            if td["done"].all():
                break

        # Collect results
        out = {
            "reward": env.get_reward(td),
            "log_likelihood": torch.stack(log_probs, dim=1).sum(dim=1),
        }

        if return_actions:
            out["actions"] = torch.stack(actions, dim=1)

        # Unbatch if multiple starts
        if num_starts > 1:
            out["reward"] = unbatchify(out["reward"], num_starts)
            out["log_likelihood"] = unbatchify(out["log_likelihood"], num_starts)
            if return_actions:
                out["actions"] = unbatchify(out["actions"], num_starts)

        return out
