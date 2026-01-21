"""
DR-GRPO (GRPO done Right) algorithm.
"""
from __future__ import annotations

import torch

from logic.src.pipeline.rl.ppo import PPO


class DRGRPO(PPO):
    """
    DR-GRPO (GRPO done Right).

    Features:
    - Zero-mean centered advantages without std normalization: A = R - Mean(R_group).
    - No sequence length normalization in the objective function.
    """

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        DR-GRPO centered advantages.
        """
        # GRPO uses group mean as baseline
        # In this implementation, we treat the batch as the group
        advantage = rewards - rewards.mean()
        return advantage

    def calculate_ratio(self, new_log_p: torch.Tensor, old_log_p: torch.Tensor) -> torch.Tensor:
        """
        Importance ratio (no sequence length normalization).
        """
        return torch.exp(new_log_p - old_log_p.detach())
