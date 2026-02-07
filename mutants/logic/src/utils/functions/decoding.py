"""
Decoding strategies for constructive policies.

This module provides various decoding strategies for autoregressive
neural network policies, including greedy, sampling, and beam search.


Reference: RL4CO (https://github.com/ai4co/rl4co)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict


class DecodingStrategy(ABC):
    """
    Abstract base class for decoding strategies.

    Decoding strategies control how actions are selected from
    the logits produced by the policy decoder.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        tanh_clipping: float = 0.0,
        mask_logits: bool = True,
        multistart: bool = False,
        num_starts: int = 1,
        select_best: bool = False,
        **kwargs,
    ):
        """
        Initialize DecodingStrategy.

        Args:
            temperature: Temperature for softmax scaling. Higher = more random.
            top_k: Keep only top-k logits before sampling.
            top_p: Keep smallest set of logits with cumsum >= top_p (nucleus sampling).
            tanh_clipping: Apply tanh clipping to logits (Bello et al., 2016).
            mask_logits: Whether to apply masking to invalid actions.
            multistart: Whether to use multiple starting points.
            num_starts: Number of starting points for multistart.
            select_best: Whether to select best solution from multiple starts.
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.multistart = multistart
        self.num_starts = num_starts
        self.select_best = select_best

    @abstractmethod
    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action from logits.

        Args:
            logits: Action logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]
            td: Optional TensorDict with additional state

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        raise NotImplementedError

    def pre_decoder_hook(
        self,
        td: TensorDict,
        env: Any,
    ) -> Tuple[TensorDict, Any, int]:
        """
        Hook called before decoding starts.

        Used for multistart expansion.

        Args:
            td: TensorDict with problem instance
            env: Environment

        Returns:
            Tuple of (td, env, num_starts)
        """
        num_starts = 1
        if self.multistart and self.num_starts > 1:
            td = batchify(td, self.num_starts)
            num_starts = self.num_starts
        return td, env, num_starts

    def post_decoder_hook(
        self,
        td: TensorDict,
        env: Any,
        log_likelihood: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, Any]:
        """
        Hook called after decoding completes.

        Used for best selection from multistart.

        Args:
            td: TensorDict with final state
            env: Environment
            log_likelihood: Log probabilities [batch, ...]
            actions: Selected actions [batch, seq_len]

        Returns:
            Tuple of (log_likelihood, actions, td, env)
        """
        if self.select_best and self.multistart:
            log_likelihood, actions, td = self._select_best(td, log_likelihood, actions, self.num_starts)
        return log_likelihood, actions, td, env

    def _select_best(
        self,
        td: TensorDict,
        log_likelihood: torch.Tensor,
        actions: torch.Tensor,
        num_starts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select best solution from multiple starts."""
        reward = td.get("reward", None)
        if reward is None:
            # Use log_likelihood as proxy
            reward = log_likelihood

        # Reshape to [batch // num_starts, num_starts]
        batch_size = reward.shape[0] // num_starts
        reward = reward.view(batch_size, num_starts)

        # Get best indices [batch_size]
        best_idx = reward.argmax(dim=1)

        # Gather best
        actions = actions.view(batch_size, num_starts, -1)
        # best_idx is [batch_size], we need [batch_size, 1, seq_len]
        gather_idx = best_idx.view(batch_size, 1, 1).expand(-1, -1, actions.shape[-1])
        actions = actions.gather(1, gather_idx)
        actions = actions.squeeze(1)

        log_likelihood = log_likelihood.view(batch_size, num_starts)
        log_likelihood = log_likelihood.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)

        # Unbatchify td
        td = unbatchify(td, num_starts)

        return log_likelihood, actions, td

    def _process_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process logits with masking, clipping, and temperature.

        Args:
            logits: Raw logits [batch, num_nodes]
            mask: Valid action mask [batch, num_nodes]

        Returns:
            Processed logits
        """
        # Apply tanh clipping (Bello et al., 2016)
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)

        # Apply mask
        if self.mask_logits:
            logits = logits.masked_fill(~mask, float("-inf"))

        # Apply temperature
        if self.temperature != 1.0:
            logits = logits / self.temperature

        # Apply top-k filtering
        if self.top_k is not None and self.top_k > 0:
            logits = top_k_filter(logits, self.top_k)

        # Apply top-p (nucleus) filtering
        if self.top_p is not None and 0 < self.top_p < 1:
            logits = top_p_filter(logits, self.top_p)

        return logits


class Greedy(DecodingStrategy):
    """Greedy decoding: always select the action with highest probability."""

    def __init__(self, **kwargs):
        """
        Initialize Greedy decoding.

        Args:
            **kwargs: Passed to super class.
        """
        # Greedy ignores temperature and sampling parameters
        super().__init__(temperature=1.0, top_k=None, top_p=None, **kwargs)

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action with highest probability."""
        logits = self._process_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)

        action = probs.argmax(dim=-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)

        # Entropy for greedy is 0 since it's deterministic
        entropy = torch.zeros_like(log_prob)

        return action, log_prob, entropy


class Sampling(DecodingStrategy):
    """Sampling decoding: sample from the probability distribution."""

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from distribution."""
        logits = self._process_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)

        # Handle numerical issues
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Calculate entropy = -sum(p * log(p))
        entropy = dist.entropy()

        return action, log_prob, entropy


class BeamSearch(DecodingStrategy):
    """
    Beam search decoding: maintain top-k partial solutions.

    Note: Beam search requires special handling in the policy forward pass.
    This implementation provides the step function for scoring candidates.
    """

    def __init__(self, beam_width: int = 5, **kwargs):
        """
        Initialize BeamSearch decoding.

        Args:
            beam_width: Number of beams to maintain.
            **kwargs: Passed to super class.
        """
        super().__init__(**kwargs)
        self.beam_width = beam_width

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k actions for beam search.

        Returns:
            Tuple of (top_k_actions, top_k_log_probs, entropy) each with shape [batch, beam_width]
        """
        logits = self._process_logits(logits, mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get top-k
        top_log_probs, top_actions = torch.topk(log_probs, self.beam_width, dim=-1)

        # Placeholder entropy for beam search (complex to define per step)
        entropy = torch.zeros_like(top_log_probs)

        return top_actions, top_log_probs, entropy


class Evaluate(DecodingStrategy):
    """
    Evaluate decoding: compute log probability of given actions.

    Used for evaluating pre-computed solutions.
    """

    def __init__(self, actions: Optional[torch.Tensor] = None, **kwargs):
        """
        Initialize Evaluate decoding.

        Args:
            actions: Pre-defined actions to evaluate [batch, seq_len].
            **kwargs: Passed to super class.
        """
        super().__init__(**kwargs)
        self.actions = actions
        self.step_idx = 0

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get log probability of pre-specified action."""
        if self.actions is None:
            raise ValueError("Actions must be provided for Evaluate strategy")

        logits = self._process_logits(logits, mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get action for current step
        action = self.actions[:, self.step_idx]
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        self.step_idx += 1

        # Entropy is 0 for evaluation
        entropy = torch.zeros_like(log_prob)

        return action, log_prob, entropy

    def reset(self):
        """Reset step counter."""
        self.step_idx = 0


# ============================================================================
# Utility Functions
# ============================================================================


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: Logits tensor [batch, num_nodes]
        k: Number of top values to keep

    Returns:
        Filtered logits with others set to -inf
    """
    if k <= 0:
        return logits

    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keep the smallest set of logits whose cumulative probability >= p.

    Args:
        logits: Logits tensor [batch, num_nodes]
        p: Cumulative probability threshold

    Returns:
        Filtered logits with others set to -inf
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Shift indices to the right to keep the first token that crosses the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original ordering
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def batchify(td: TensorDict, num_repeats: int) -> TensorDict:
    """
    Repeat TensorDict for multistart.

    Args:
        td: TensorDict [batch, ...]
        num_repeats: Number of times to repeat

    Returns:
        TensorDict [batch * num_repeats, ...]
    """
    # Use interleave to keep instances together
    # Use interleave to keep instances together
    # Some TensorDict versions might not have repeat_interleave on the object
    try:
        return td.repeat_interleave(num_repeats, dim=0)
    except AttributeError:
        # Fallback: repeat manually for each key
        # We must set batch_size FIRST to allow item assignment of different shape
        new_batch_size = torch.Size([td.batch_size[0] * num_repeats, *td.batch_size[1:]])
        new_td = TensorDict({}, batch_size=new_batch_size, device=td.device)
        for key, val in td.items():
            if isinstance(val, torch.Tensor):
                new_td[key] = val.repeat_interleave(num_repeats, dim=0)
        return new_td


def unbatchify(td: TensorDict, num_repeats: int) -> TensorDict:
    """
    Reverse batchify: take first of each group.

    Args:
        td: TensorDict [batch * num_repeats, ...]
        num_repeats: Number of repeats used

    Returns:
        TensorDict [batch, ...]
    """
    batch_size = td.batch_size[0] // num_repeats
    indices = torch.arange(0, batch_size * num_repeats, num_repeats, device=td.device)
    return td[indices]


def gather_by_index(src: torch.Tensor, idx: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Gather elements from src using indices.

    Args:
        src: Source tensor
        idx: Index tensor
        dim: Dimension to gather along

    Returns:
        Gathered tensor
    """
    idx = idx.unsqueeze(-1).expand(*idx.shape, src.shape[-1])
    return src.gather(dim, idx)


# ============================================================================
# Strategy Registry
# ============================================================================


DECODING_STRATEGY_REGISTRY = {
    "greedy": Greedy,
    "sampling": Sampling,
    "beam_search": BeamSearch,
    "evaluate": Evaluate,
}


def get_decoding_strategy(
    name: str,
    **kwargs,
) -> DecodingStrategy:
    """
    Get decoding strategy by name.

    Args:
        name: Strategy name ('greedy', 'sampling', 'beam_search', 'evaluate')
        **kwargs: Strategy-specific parameters

    Returns:
        Initialized decoding strategy
    """
    name = name.lower()
    if name not in DECODING_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown decoding strategy: {name}. " f"Available: {list(DECODING_STRATEGY_REGISTRY.keys())}")
    return DECODING_STRATEGY_REGISTRY[name](**kwargs)


__all__ = [
    "DecodingStrategy",
    "Greedy",
    "Sampling",
    "BeamSearch",
    "Evaluate",
    "get_decoding_strategy",
    "top_k_filter",
    "top_p_filter",
    "batchify",
    "unbatchify",
    "gather_by_index",
]
