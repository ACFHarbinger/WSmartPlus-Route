"""
Autoregressive Policy Base Classes.

This module provides base classes for autoregressive (AR) constructive models.
AR models build solutions step-by-step by selecting one action at a time,
where each action depends on the previous actions and the encoded state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.common.constructive import ConstructivePolicy
from tensordict import TensorDict


class AutoregressiveEncoder(nn.Module, ABC):
    """
    Base class for autoregressive encoders.

    AR encoders take a problem instance and produce initial embeddings
    that represent the problem state.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize AutoregressiveEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Compute initial embeddings for the problem instance.

        Args:
            td: TensorDict containing problem instance.

        Returns:
            Embeddings tensor or tuple of tensors representing the encoded state.
        """
        raise NotImplementedError


class AutoregressiveDecoder(nn.Module, ABC):
    """
    Base class for autoregressive decoders.

    AR decoders select actions sequentially until a termination condition is met.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize AutoregressiveDecoder."""
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
        Construct solutions step-by-step.

        Args:
            td: TensorDict containing problem instance.
            embeddings: Encoded state from the encoder.
            env: Environment for state transitions.

        Returns:
            Tuple of (log_likelihood, actions).
        """
        raise NotImplementedError


class AutoregressivePolicy(ConstructivePolicy):
    """
    Base class for autoregressive policies.

    Combines an AR encoder with an AR decoder to form a complete policy.
    Inherits from ConstructivePolicy to leverage standardized decoding strategies.
    """

    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize AutoregressivePolicy."""
        # Use placeholders for base initialization if not provided
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            embed_dim=embed_dim,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        decode_type: str = "sampling",
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode + decode.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions.
            decode_type: Decoding strategy.
            num_starts: Number of solution starts.

        Returns:
            Dictionary containing reward, log_likelihood, and actions.
        """
        # Encode
        embeddings = self.encoder(td, **kwargs) if self.encoder is not None else None

        # Decode
        if self.decoder is not None:
            # Note: Many decoders in WSmart-Route implement their own loop.
            # We assume the decoder handles the AR process.
            log_p, actions = self.decoder(td, embeddings, env, decode_type=decode_type, num_starts=num_starts, **kwargs)
        else:
            raise ValueError("AutoregressivePolicy requires a decoder.")

        # Calculate reward
        reward = env.get_reward(td, actions)

        return {
            "reward": reward,
            "log_likelihood": log_p,
            "actions": actions,
        }
