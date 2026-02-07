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
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce logits and mask for the current step.

        Args:
            td: TensorDict containing current state.
            heatmap: Pre-computed heatmap from encoder.
            env: Environment.

        Returns:
            Tuple of (logits, mask).
        """
        raise NotImplementedError

    def construct(
        self,
        td: TensorDict,
        heatmap: torch.Tensor,
        env: RL4COEnvBase,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full construction of solutions from heatmaps.
        Default implementation returns empty dict, subclasses should override.
        """
        return {}


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
        encoder_out = self.encoder(td, **kwargs) if self.encoder is not None else None
        if isinstance(encoder_out, tuple):
            heatmap = encoder_out[0]
        else:
            heatmap = encoder_out

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

    def common_decoding(
        self,
        decode_type: str,
        td: TensorDict,
        env: RL4COEnvBase,
        heatmap: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **decoding_kwargs,
    ):
        """
        Common decoding logic for NAR models.

        Args:
            decode_type: Decoding strategy ('sampling', 'greedy', etc.)
            td: Initial TensorDict
            env: Environment
            heatmap: Predicted heatmap from encoder
            actions: Pre-specified actions for evaluation
            **decoding_kwargs: Additional arguments for decoding

        Returns:
            Tuple of (logprobs, actions, td, env)
        """
        from logic.src.utils.functions.decoding import get_decoding_strategy

        if actions is not None:
            decode_type = "evaluate"
            decoding_kwargs["actions"] = actions

        strategy = get_decoding_strategy(decode_type, **decoding_kwargs)

        # Pre-decoding hook
        td, env, num_starts_hook = strategy.pre_decoder_hook(td, env)

        # Determine num_starts
        num_starts = decoding_kwargs.get("num_starts", decoding_kwargs.get("num_samples", num_starts_hook))
        if num_starts is None:
            num_starts = 1
        num_starts = int(num_starts)

        # Update heatmap and td to match num_starts if needed
        if num_starts > 1:
            from logic.src.utils.functions.decoding import batchify

            if td.size(0) != heatmap.size(0) * num_starts:
                # This might happen if pre_decoder_hook didn't batchify
                if td.size(0) == heatmap.size(0):
                    td = batchify(td, num_starts)

            if heatmap.size(0) != td.size(0):
                heatmap = batchify(heatmap, num_starts)

        if self.decoder is None:
            raise ValueError("common_decoding requires a decoder to be set")

        # Main decoding loop
        step = 0
        actions_list = []
        log_probs_list = []
        while not td["done"].all():
            # In NAR models, the decoder uses the pre-computed heatmap
            logits, mask = self.decoder(td, heatmap, env)
            action, log_prob, _ = strategy.step(logits, mask, td)

            actions_list.append(action)
            log_probs_list.append(log_prob)

            td.set("action", action)
            td.update(env.step(td)["next"])
            step += 1

        actions = torch.stack(actions_list, dim=1)
        logprobs = torch.stack(log_probs_list, dim=1)

        # Post-decoding hook
        logprobs, actions, td, env = strategy.post_decoder_hook(td, env, logprobs, actions)
        return logprobs, actions, td, env

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        return self
