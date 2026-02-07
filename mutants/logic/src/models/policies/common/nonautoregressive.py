"""
Non-Autoregressive Policy Base Classes.

This module provides base classes for non-autoregressive (NAR) constructive models.
NAR models predict edge/node heatmaps (log probabilities) in a single forward pass,
then construct solutions using these heatmaps (e.g., via ACO, beam search, greedy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from tensordict import TensorDict


class NonAutoregressiveEncoder(nn.Module, ABC):
    """
    Base class for non-autoregressive encoders.

    NAR encoders take a problem instance and produce heatmaps
    (log probabilities over edges or nodes) in a single forward pass.
    """

    def __init__(self, embed_dim: int = 128, **kwargs):
        """Initialize NonAutoregressiveEncoder."""
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute heatmap logits for the problem instance.

        Args:
            td: TensorDict containing problem instance (locs, demands, etc.)

        Returns:
            Heatmap tensor of shape [batch, num_nodes, num_nodes] for edge-based,
            or [batch, num_nodes] for node-based.
        """
        raise NotImplementedError


class NonAutoregressiveDecoder(nn.Module, ABC):
    """
    Base class for non-autoregressive decoders.

    NAR decoders construct solutions from heatmaps using various methods
    like Ant Colony Optimization, greedy decoding, or sampling.
    """

    def __init__(self, **kwargs):
        """Initialize NonAutoregressiveDecoder."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Construct solutions from heatmaps.

        Args:
            td: TensorDict containing problem instance.
            heatmap: Heatmap tensor from encoder.
            env: Environment for solution validation.

        Returns:
            Dictionary containing:
                - actions: Solution sequence [batch, seq_len]
                - log_likelihood: Log probability of solutions [batch]
                - reward: Reward/cost of solutions [batch]
        """
        raise NotImplementedError


class NonAutoregressivePolicy(nn.Module, ABC):
    """
    Base class for non-autoregressive policies.

    Combines a NAR encoder (heatmap prediction) with a NAR decoder
    (solution construction) to form a complete policy.
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: Optional[str] = None,
        embed_dim: int = 128,
        **kwargs,
    ):
        """Initialize NonAutoregressivePolicy."""
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.env_name = env_name
        self.embed_dim = embed_dim

    def forward(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
        num_starts: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full forward pass: encode heatmap + decode solution.

        Args:
            td: TensorDict containing problem instance.
            env: Environment for state transitions and reward calculation.
            num_starts: Number of solution constructions (for stochastic decoders).
            **kwargs: Additional arguments for encoder/decoder.

        Returns:
            Dictionary containing:
                - reward: Final reward/cost [batch] or [batch, num_starts]
                - log_likelihood: Log probability of solution [batch]
                - actions: Solution sequence [batch, seq_len]
                - heatmap: Predicted heatmap from encoder
        """
        # Encode: predict heatmap
        heatmap = self.encoder(td, **kwargs) if self.encoder is not None else None

        # Decode: construct solution(s) from heatmap
        if self.decoder is not None and heatmap is not None:
            out = self.decoder(td, heatmap, env, num_starts=num_starts, **kwargs)
        else:
            # Fallback for subclasses that override forward entirely
            out = {}

        out["heatmap"] = heatmap
        return out

    def set_decode_type(self, decode_type: str, **kwargs):
        """Set decode type (compatibility with evaluation pipeline)."""
        self._decode_type = decode_type
        for k, v in kwargs.items():
            setattr(self, f"_{k}", v)

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self
