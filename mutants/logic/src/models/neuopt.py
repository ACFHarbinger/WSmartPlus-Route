"""
Neural Optimizer (NeuOpt) Model.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn
from logic.src.envs.base import RL4COEnvBase
from logic.src.models.policies.neuopt import NeuOptPolicy


class NeuOpt(nn.Module):
    """
    NeuOpt Model: Neural Optimizer for combinatorial optimization.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[NeuOptPolicy] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **policy_kwargs,
    ):
        super().__init__()
        if policy is None:
            policy = NeuOptPolicy(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                **policy_kwargs,
            )
        self.policy = policy
        self.env = env

    def forward(
        self,
        td: Any,
        env: Optional[RL4COEnvBase] = None,
        decode_type: str = "greedy",
        **kwargs,
    ):
        """Forward pass of the model."""
        env = env or self.env
        return self.policy(td, env, decode_type=decode_type, **kwargs)
