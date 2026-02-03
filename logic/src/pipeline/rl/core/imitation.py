"""
Imitation Learning algorithm implementation.
Trains a policy to mimic an expert solver (e.g., HGS, Local Search).
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.utils.data.rl_utils import safe_td_copy


class ImitationLearning(RL4COLitModule):
    """
    Imitation Learning / Supervised Learning.

    Minimizes the difference between the policy's predicted action probabilities
    and the actions taken by an expert solver (e.g., HGS, Local Search).

    Features:
    - On-the-fly expert data generation or pre-computed dataset support.
    - Cross Entropy Loss.
    - Support for multiple expert types.
    """

    def __init__(
        self,
        expert_policy: Any = None,  # Policy or Solver class
        expert_name: str = "hgs",
        **kwargs,
    ):
        """
        Initialize ImitationLearning module.

        Args:
            expert_policy: Expert policy or solver to imitate.
            expert_name: Name of expert solver ('hgs', 'local_search').
            **kwargs: Arguments passed to RL4COLitModule.
        """
        # Baseline is not used in IL, but we keep the structure
        kwargs["baseline"] = "none"
        super().__init__(**kwargs)
        self.expert_policy = expert_policy
        self.expert_name = expert_name

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Compute Cross Entropy Loss between Policy and Expert.
        """
        # 1. Generate Expert Solutions
        with torch.no_grad():
            if self.expert_name == "hgs":
                # Assuming expert_policy is an HGSPolicy instance or wrapper
                expert_out = self.expert_policy(td, self.env)
                expert_actions = expert_out["actions"]  # [batch, seq_len]
            elif self.expert_name in ["local_search", "random_ls"]:
                # Assuming expert_policy is an instance of RandomLocalSearchPolicy
                expert_out = self.expert_policy(td, self.env)
                expert_actions = expert_out["actions"]
            else:
                raise ValueError(f"Unknown expert: {self.expert_name}")

        # 2. Re-evaluate Policy to force expert actions (teacher forcing)?
        # Constructive policies in RL4CO output log_probs for the TAKEN actions.
        # If we just want to train BCl (Behavior Cloning), we usually:
        # a) Feed expert actions as input (teacher forcing) if model supports it.
        # b) Compute log_prob of expert actions.

        # RL4CO models support `actions=expert_actions` to return log_prob of those actions.
        # We need to call the policy again with the expert actions.

        # Reset env to ensure clean state
        td_clone = safe_td_copy(td)
        td_clone = self.env.reset(td_clone)

        # Teacher Forcing: Evaluate log likelihood of expert actions
        # Note: Constructive policies output 'log_likelihood' summed over sequence usually,
        # or log_probs per step. We need standard Cross Entropy or NLL.
        # Actually, "log_likelihood" output IS sum(log p(action)).
        # Maximizing this is equivalent to minimizing NLL.
        # Loss = -mean(log P(expert_action|state))

        teacher_out = self.policy(td_clone, self.env, actions=expert_actions)
        log_likelihood = teacher_out["log_likelihood"]

        loss = -log_likelihood.mean()

        # Log accuracy/matching? harder for constructive
        self.log("train/loss", loss, prog_bar=True)

        return loss
