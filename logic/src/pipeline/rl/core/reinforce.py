"""
REINFORCE algorithm implementation.

Reference:
    Williams, R. J. (1992).
    Simple statistical gradient-following algorithms for connectionist reinforcement learning.
    Machine learning, 8(3-4), 229-256.

Attributes:
    REINFORCE: REINFORCE algorithm.

Example:
    >>> from logic.src.pipeline.rl.core import REINFORCE
    >>> from logic.src.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> reinforce = REINFORCE(env, agent)
    >>> reinforce
    REINFORCE(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from typing import TYPE_CHECKING, Optional

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from logic.src.interfaces.env import IEnv

from logic.src.pipeline.rl.common.base import RL4COLitModule


class REINFORCE(RL4COLitModule):
    """
    REINFORCE with baseline.

    Standard policy gradient algorithm with configurable baselines.

    Reference:
        Williams, R. J. (1992).
        Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
        Machine Learning, 8(3-4), 229-256.
        https://doi.org/10.1007/BF00992696

    Attributes:
        policy: Policy network.
        critic: Critic network.
        entropy_weight: Weight for entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
    """

    def __init__(
        self,
        entropy_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        lr_critic: float = 1e-4,
        **kwargs,
    ):
        """
        Initialize REINFORCE module.

        Args:
            entropy_weight: Weight for the entropy bonus (default: 0.0).
            max_grad_norm: Maximum gradient norm for clipping (default: 1.0).
            lr_critic: Learning rate for the critic optimizer (default: 1e-4).
            kwargs: Additional arguments to pass to the parent class (RL4COLitModule).
        """
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.max_grad_norm = max_grad_norm

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional["IEnv"] = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss.

        Loss = -E[(R - b) * log π(a|s)]

        Args:
            td: Input tensor dictionary containing state information.
            out: Output dictionary from the policy containing 'reward', 'log_likelihood', and optionally 'entropy'.
            batch_idx: Index of the current batch.
            env: Environment instance for baseline evaluation (optional).

        Returns:
            Computed policy gradient loss tensor.
        """
        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Get baseline
        if hasattr(self, "_current_baseline_val") and self._current_baseline_val is not None:
            baseline_val = self._current_baseline_val
        else:
            baseline_val = self.baseline.eval(td, reward, env=env)

        # Advantage
        advantage = reward - baseline_val

        # Normalize advantage (optional but helps stability)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy gradient loss
        loss = -(advantage.detach() * log_likelihood).mean()

        # Log loss components
        self.log("train/policy_loss", loss, sync_dist=True)
        self.log("train/log_likelihood", log_likelihood.mean(), sync_dist=True)

        # Entropy bonus (if applicable)
        if self.entropy_weight > 0 and "entropy" in out:
            loss = loss - self.entropy_weight * out["entropy"].mean()
            self.log("train/entropy", out["entropy"].mean(), sync_dist=True)

        # Log components
        self.log("train/advantage", advantage.mean(), sync_dist=True)
        self.log("train/baseline", baseline_val.mean(), sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping.

        Args:
            optimizer: PyTorch optimizer instance about to step.
        """
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),  # type: ignore[attr-defined]
                self.max_grad_norm,
            )
