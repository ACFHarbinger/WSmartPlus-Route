"""
N-step Proximal Policy Optimization (PPO).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.pipeline.rl.core.ppo import PPO


class PPOStep(PPO):
    """
    N-step PPO algorithm.

    Performs PPO updates using N-step returns for better credit assignment.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: nn.Module,
        n_steps: int = 5,
        gamma: float = 0.99,
        **kwargs,
    ):
        """
        Initialize N-step PPO.

        Args:
            env: RL environment.
            policy: Actor network.
            critic: Critic network.
            n_steps: Number of lookahead steps.
            gamma: Discount factor.
            **kwargs: Passed to PPO super class.
        """
        super().__init__(env=env, policy=policy, critic=critic, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[RL4COEnvBase] = None,
    ) -> torch.Tensor:
        """
        Calculate N-step PPO loss.
        """
        # Placeholder for N-step implementation
        # Real implementation requires trajectory collection which differs from standard PPO
        # For now, we reuse PPO's loss but this would be extended in full implementation
        return super().calculate_loss(td, out, batch_idx, env)


__all__ = ["PPOStep"]
