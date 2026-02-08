"""
L2D Model: Learning to Dispatch.

Wrapper for L2DPolicy to be used in RL pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase

from .policy import L2DPolicy


class L2DModel(nn.Module):
    """
    L2D Model for Job Shop Scheduling.

    Wraps L2DPolicy for training and inference.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[L2DPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline: str = "rollout",
        **kwargs,
    ):
        """
        Initialize L2D Model.

        Args:
            env: Environment (optional).
            policy: L2DPolicy instance (optional).
            policy_kwargs: Arguments for L2DPolicy if policy is not provided.
            baseline: Baseline type for REINFORCE (e.g. "rollout", "mean").
        """
        super().__init__()
        self.env = env

        if policy is None:
            policy_kwargs = policy_kwargs or {}
            self.policy = L2DPolicy(env_name=env.name if env else "jssp", **policy_kwargs)
        else:
            self.policy = policy

        self.baseline = baseline

    def forward(self, td: TensorDict, phase: str = "train", return_actions: bool = False, **kwargs) -> dict:
        """Forward pass forwarding to policy."""
        return self.policy(td, self.env, phase=phase, return_actions=return_actions, **kwargs)
