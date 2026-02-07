"""
L2D PPO Model: Learning to Dispatch with PPO.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.critic import CriticNetwork
from logic.src.models.policies.l2d import L2DPolicy, L2DPolicy4PPO
from logic.src.pipeline.rl.core.stepwise_ppo import StepwisePPO


class L2DPPOModel(StepwisePPO):
    """
    L2D Model trained with Stepwise PPO.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[L2DPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        critic: Optional[nn.Module] = None,
        critic_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize L2DPPOModel (StepwisePPO).

        Args:
            env: Environment instance (optional).
            policy: L2DPolicy (or L2DPolicy4PPO) instance.
            policy_kwargs: Arguments for policy.
            critic: Critic network instance.
            critic_kwargs: Arguments for critic network.
            **kwargs: Additional arguments for StepwisePPO.
        """
        if env is None:
            # Default to JSSP
            from logic.src.envs import JSSPEnv

            env = JSSPEnv()

        if policy is None:
            policy_kwargs = policy_kwargs or {}
            policy = L2DPolicy4PPO(env_name=env.name, **policy_kwargs)

        if critic is None:
            critic_kwargs = critic_kwargs or {}
            critic = CriticNetwork(env_name=env.name, **critic_kwargs)

        super().__init__(env=env, policy=policy, critic=critic, **kwargs)
