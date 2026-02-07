"""
Probabilistic Operations (Entropy, etc).
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
