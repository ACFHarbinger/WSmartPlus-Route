"""
Sampling decoding strategy implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.constants.models import NUMERICAL_EPSILON

from .base import DecodingStrategy


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
        probs = probs.clamp(min=NUMERICAL_EPSILON)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Calculate entropy = -sum(p * log(p))
        entropy = dist.entropy()

        return action, log_prob, entropy
