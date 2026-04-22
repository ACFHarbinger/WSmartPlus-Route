"""
Probabilistic Operations (Entropy, etc).

Attributes:
    calculate_entropy: Calculate entropy from log probabilities.

Example:
    >>> from logic.src.utils.ops.probabilistic import calculate_entropy
    >>> logprobs = torch.randn(10, 5)
    >>> entropy = calculate_entropy(logprobs)
    >>> print(entropy.shape)
    torch.Size([10])
"""

from __future__ import annotations

import torch


def calculate_entropy(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy from log probabilities.

    H = -sum(p * log(p))

    Args:
        logprobs: Log probabilities [batch, ..., n_actions].

    Returns:
        Entropy [batch, ...].
    """
    probs = logprobs.exp()
    entropy = -(probs * logprobs).sum(dim=-1)
    return entropy
