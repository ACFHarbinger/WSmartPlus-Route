"""
Sampling decoding strategy implementation.

Attributes:
    Sampling: Sampling decoding strategy.

Example:
    >>> from logic.src.utils.decoding import Sampling
    >>> strategy = Sampling()
    >>> strategy.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[True, True], [True, True]]))
    (tensor([1, 1]), tensor([0.2, 0.4]), tensor([0.2, 0.4]))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.constants.models import NUMERICAL_EPSILON

from .base import DecodingStrategy


class Sampling(DecodingStrategy):
    """
    Sampling decoding: sample from the probability distribution.

    Attributes:
        temperature: Temperature for sampling.
        top_k: Top-k filtering.
        top_p: Top-p filtering.
    """

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from distribution.

        Args:
            logits: Logits tensor [batch, num_nodes]
            mask: Mask tensor [batch, num_nodes]
            td: TensorDict (unused)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
        logits = self._process_logits(logits, mask)
        probs = F.softmax(logits, dim=-1)

        # Handle numerical issues
        probs = probs.clamp(min=NUMERICAL_EPSILON)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Calculate entropy = -sum(p * log(p))
        entropy = dist.entropy()

        return action, log_prob, entropy
