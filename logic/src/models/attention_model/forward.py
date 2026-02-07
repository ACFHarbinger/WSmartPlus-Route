"""
Forward pass logic for AttentionModel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.utils.functions.beam_search import CachedLookup


class ForwardMixin:
    """Mixin for forward pass logic."""

    def __init__(self):
        # Type hints
        self.problem: Any
        self.context_embedder: nn.Module
        self.encoder: nn.Module
        self.decoder: nn.Module
        self.project_node_embeddings: nn.Module
        self.aggregation_graph: str
        self.checkpoint_encoder: bool
        self.decode_type: str
        self.temp: float
        self.pomo_size: int

    def _get_initial_embeddings(self, input: Dict[str, torch.Tensor]):
        """
        Get initial node embeddings from the context embedder.

        Args:
            input: The input data dictionary.

        Returns:
            Initial node embeddings.
        """
        return self.context_embedder(input)

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        env: Optional[Any] = None,
        decode_type: Optional[str] = None,
        return_pi: bool = False,
        pad: bool = False,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward pass of the Attention Model.

        Args:
            input: Problem state dictionary/TensorDict.
            env: Environment instance.
            decode_type: Decoding strategy ('greedy' or 'sampling').
            return_pi: Whether to return the policy log probabilities.
            pad: Whether to pad keys for non-square matrices (unused here, kept for API compat).
            mask: Optional mask for valid nodes.
            expert_pi: Optional expert policy for imitation learning.
            **kwargs: Additional arguments.

        Returns:
            Dictionary containing 'reward', 'cost', 'log_likelihood', and optionally 'pi'.
        """
        # Ensure embedding init
        if not hasattr(self, "project_node_embeddings"):
            # Should have been called in __init__, but safe guard
            pass

        # Use efficient embeddings
        embeddings, init_context = self._get_initial_embeddings(input)

        # Pass through Encoder
        if self.checkpoint_encoder and self.training:
            # Gradient checkpointing for memory efficiency
            outputs = torch.utils.checkpoint.checkpoint(self.encoder, embeddings, mask, use_reentrant=False)
        else:
            outputs = self.encoder(embeddings, mask)

        # Graph aggregation for context
        # (batch, seq, hidden) -> (batch, hidden)
        if self.aggregation_graph == "avg":
            graph_context = outputs.mean(1)
        elif self.aggregation_graph == "max":
            graph_context = outputs.max(1)[0]
        elif self.aggregation_graph == "sum":
            graph_context = outputs.sum(1)
        else:
            # Default to average
            graph_context = outputs.mean(1)

        # Pass through Decoder
        # The decoder handles autoregressive steps
        _log_p, pi, cost = self.decoder(
            input,
            outputs,
            graph_context,
            init_context,
            env,
            decode_type=decode_type or self.decode_type,
            return_pi=return_pi,
            expert_pi=expert_pi,
        )

        out = {"cost": cost, "reward": -cost}  # RL maximizes reward

        if _log_p is not None:
            out["log_likelihood"] = _log_p

        if return_pi and pi is not None:
            out["pi"] = pi

        return out

    def precompute_fixed(self, input: Dict[str, torch.Tensor], edges: Optional[torch.Tensor]):
        """
        Precompute fixed embeddings for the input.

        Args:
            input: The input data.
            edges: Edge information for the graph.

        Returns:
            CachedLookup: A cached lookup object containing precomputed decoder state.
        """
        embeddings, init_context = self._get_initial_embeddings(input)
        _log_p, _pi, _cost = self.decoder(input, embeddings, init_context, None, precompute_only=True)

        # Return a lookup object compatible with beam search
        # Note: The exact structure depends on what beam_search expects
        # This is a simplified placeholder based on typical usage
        return CachedLookup(embeddings=embeddings, context=init_context)

    def expand(self, t):
        """
        Expand tensor or dictionary of tensors for POMO.

        Args:
            t (torch.Tensor or dict or None): Input to expand.

        Returns:
            Expanded input.
        """
        if t is None:
            return None
        if isinstance(t, dict) or isinstance(t, TensorDict):
            return t.__class__({k: self.expand(v) for k, v in t.items()})

        # Expand (Batch, ...) -> (Batch * POMO, ...)
        # We repeat the batch elements
        bs = t.size(0)
        shape = (bs, self.pomo_size) + t.shape[1:]
        return t.unsqueeze(1).expand(shape).reshape(-1, *t.shape[1:])
