"""
Stepwise Proximal Policy Optimization (PPO).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.pipeline.rl.core.ppo import PPO


class PPOStepwise(PPO):
    """
    Stepwise PPO algorithm.

    Performs PPO updates at each step of decoding rather than trajectory level.
    Ideal for MDP environments where rewards are dense.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: nn.Module,
        **kwargs,
    ):
        super().__init__(env=env, policy=policy, critic=critic, **kwargs)

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Optional[RL4COEnvBase] = None,
    ) -> torch.Tensor:
        """
        Calculate stepwise PPO loss.
        """
        # Placeholder for stepwise implementation
        # Standard PPO implementation in this repo is already trajectory-based
        # Stepwise requires per-step advantages and dense rewards
        return super().calculate_loss(td, out, batch_idx, env)


__all__ = ["PPOStepwise"]
