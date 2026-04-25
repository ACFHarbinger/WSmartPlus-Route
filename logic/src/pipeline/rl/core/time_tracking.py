"""Time-aware Reinforcement Learning training.

This algorithm optimizes for solution generation time by incorporating time
as a factor in the reward signal.

Attributes:
    TimeOptimizedREINFORCE: REINFORCE variation that optimizes for solution generation time.

Example:
    >>> from logic.src.pipeline.rl.core import TimeOptimizedREINFORCE
    >>> from logic.src.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> time_optimized_reinforce = TimeOptimizedREINFORCE(env, agent)
    >>> time_optimized_reinforce
    TimeOptimizedREINFORCE(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from typing import TYPE_CHECKING, Optional

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv
from logic.src.models.common.time_tracking_policy import TimeTrackingPolicy

from .reinforce import REINFORCE


class TimeOptimizedREINFORCE(REINFORCE):
    """
    REINFORCE variation that optimizes for solution generation time.

    Attributes:
        time_sensitivity: Weight for the time penalty.
    """

    def __init__(
        self,
        time_sensitivity: float = 0.0,
        **kwargs,
    ):
        """
        Initialize TimeOptimizedREINFORCE.

        Args:
            time_sensitivity: Weight for the time penalty (default: 0.0).
            kwargs: Additional arguments to pass to the parent class (REINFORCE).
        """
        super().__init__(**kwargs)
        self.time_sensitivity = time_sensitivity

        # Wrap policy for time tracking if needed
        if self.time_sensitivity > 0:
            self.policy = TimeTrackingPolicy(self.policy)  # type: ignore[arg-type, assignment]

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional["IEnv"] = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss with time penalty.

        Args:
            td: Input tensor dictionary containing state information.
            out: Dictionary containing reward and log likelihood.
            batch_idx: Index of the current batch.
            env: Environment instance.

        Returns:
            The computed loss for logging.
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Apply time penalty if tracked
        if self.time_sensitivity > 0 and "inference_time" in out:
            # Penalty = sensitivity * duration (in seconds)
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
            # Baseline typically estimates solution quality (reward)
            baseline_val = self.baseline.eval(td, reward, env=env)

        # Advantage based on effective reward
        advantage = effective_reward - baseline_val

        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy gradient loss
        loss = -(advantage.detach() * log_likelihood).mean()

        # Log loss components
        self.log("train/policy_loss", loss, sync_dist=True)
        self.log("train/log_likelihood", log_likelihood.mean(), sync_dist=True)

        # Entropy bonus
        if self.entropy_weight > 0 and "entropy" in out:
            loss = loss - self.entropy_weight * out["entropy"].mean()
            self.log("train/entropy", out["entropy"].mean(), sync_dist=True)

        # Logging
        self.log("train/advantage", advantage.mean(), sync_dist=True)
        self.log("train/baseline", baseline_val.mean(), sync_dist=True)
        self.log("train/effective_reward", effective_reward.mean(), sync_dist=True)

        return loss
