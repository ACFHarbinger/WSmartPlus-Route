"""
Greedy decoding strategy implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy


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
