"""
Evaluate decoding strategy implementation.

Attributes:
    Evaluate: Evaluate decoding strategy.

Example:
    >>> from logic.src.utils.decoding import Evaluate
    >>> strategy = Evaluate(actions=torch.tensor([[0, 1], [2, 3]]))
    >>> strategy.step(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[True, True], [True, True]]))
    (tensor([0, 2]), tensor([0.1, 0.3]), tensor([0.1, 0.3]))
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy


class Evaluate(DecodingStrategy):
    """
    Evaluate decoding: compute log probability of given actions.

    Used for evaluating pre-computed solutions.

    Attributes:
        actions: Pre-defined actions to evaluate [batch, seq_len].
        step_idx: Current step index.
    """

    def __init__(self, actions: Optional[torch.Tensor] = None, **kwargs) -> None:
        """
        Initialize Evaluate decoding.

        Args:
            actions: Pre-defined actions to evaluate [batch, seq_len].
            kwargs: Passed to super class.
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
        """
        Get log probability of pre-specified action.

        Args:
            logits: Logits tensor [batch, num_nodes]
            mask: Mask tensor [batch, num_nodes]
            td: TensorDict (unused)

        Returns:
            Tuple of (action, log_prob, entropy)
        """
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

    def reset(self) -> None:
        """
        Reset step counter.
        """
        self.step_idx = 0
