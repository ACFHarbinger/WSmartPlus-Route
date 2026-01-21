"""
REINFORCE algorithm implementation.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.base import RL4COLitModule


class REINFORCE(RL4COLitModule):
    """
    REINFORCE with baseline.

    Standard policy gradient algorithm with configurable baselines.
    """

    def __init__(
        self,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss.

        Loss = -E[(R - b) * log π(a|s)]
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Get baseline
        baseline_val = self.baseline.eval(td, reward)

        # Advantage
        advantage = reward - baseline_val

        # Normalize advantage (optional but helps stability)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy gradient loss
        loss = -(advantage.detach() * log_likelihood).mean()

        # Entropy bonus (if applicable)
        if self.entropy_weight > 0 and "entropy" in out:
            loss = loss - self.entropy_weight * out["entropy"].mean()

        # Log components
        self.log("train/advantage", advantage.mean(), sync_dist=True)
        self.log("train/baseline", baseline_val.mean(), sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
