"""
Beam Search decoding strategy.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy


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
