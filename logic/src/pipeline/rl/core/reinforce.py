"""
REINFORCE algorithm implementation.

Reference:
    Williams, R. J. (1992).
    Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    Machine learning, 8(3-4), 229-256.
"""

from typing import TYPE_CHECKING, Optional

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from logic.src.envs.base import RL4COEnvBase

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.core.time_tracking import TimeTrackingPolicy


class REINFORCE(RL4COLitModule):
    """
    REINFORCE with baseline.

    Standard policy gradient algorithm with configurable baselines.

    Now supports time-optimized training via `time_sensitivity`.

    Reference:
        Williams, R. J. (1992).
        Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
        Machine Learning, 8(3-4), 229-256.
        https://doi.org/10.1007/BF00992696
    """

    def __init__(
        self,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        lr_critic: float = 1e-4,
        time_sensitivity: float = 0.0,
        **kwargs,
    ):
        """
        Initialize REINFORCE module.

        Args:
            entropy_weight: Weight for entropy bonus in loss.
            max_grad_norm: Maximum gradient norm for clipping.
            time_sensitivity: Weight for time penalty. If > 0, penalizes slower inference.
            **kwargs: Arguments passed to RL4COLitModule.
        """
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm
        self.time_sensitivity = time_sensitivity

        # Wrap policy for time tracking if needed
        if self.time_sensitivity > 0:
            self.policy = TimeTrackingPolicy(self.policy)

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional["RL4COEnvBase"] = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss.

        Loss = -E[(R - b) * log π(a|s)]
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Apply time penalty if tracked
        if self.time_sensitivity > 0 and "inference_time" in out:
            # Penalty = sensitivity * duration (in seconds)
            # We subtract penalty from reward (Maximize Reward - Penalty)
            time_penalty = self.time_sensitivity * out["inference_time"]
            effective_reward = reward - time_penalty

            # Log time metrics
            self.log("train/inference_time", out["inference_time"].mean(), sync_dist=True)
            self.log("train/time_penalty", time_penalty.mean(), sync_dist=True)
        else:
            effective_reward = reward

        # Get baseline
        if hasattr(self, "_current_baseline_val") and self._current_baseline_val is not None:
            baseline_val = self._current_baseline_val
        else:
            # Only pass raw reward to baseline network??
            # Usually baseline estimates R. If R changes to (R-penalty),
            # we should probably train baseline on effective_reward.
            # But changing target dynamically might destabilize it.
            # For simplicity: Baseline estimates problem Reward (solution quality).
            # Time optimization is an "external" pressure on policy.
            # So Advantage = (R_eff - b_quality).
            # If policy gets faster but R drops, advantage drops.
            # If policy gets same R but faster, R_eff increases -> Positive advantage.
            baseline_val = self.baseline.eval(td, reward, env=env)

        # Advantage
        advantage = effective_reward - baseline_val

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
        self.log("train/effective_reward", effective_reward.mean(), sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
