"""
Imitation Learning algorithm implementation.
Trains a policy to mimic an expert solver (e.g., HGS, Local Search).
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.core.losses import (
    js_divergence_loss,
    kl_divergence_loss,
    nll_loss,
    reverse_kl_divergence_loss,
    weighted_nll_loss,
)
from logic.src.utils.data.rl_utils import safe_td_copy


class ImitationLearning(RL4COLitModule):
    """
    Imitation Learning / Supervised Learning.

    Minimizes the difference between the policy's predicted action probabilities
    and the actions taken by an expert solver (e.g., HGS, Local Search).

    Features:
    - On-the-fly expert data generation or pre-computed dataset support.
    - Configurable Loss (NLL, Weighted NLL, KL, JS).
    - Support for multiple expert types.
    """

    def __init__(
        self,
        expert_policy: Any = None,  # Policy or Solver class
        expert_name: str = "hgs",
        loss_fn: str = "nll",
        **kwargs,
    ):
        """
        Initialize ImitationLearning module.

        Args:
            expert_policy: Expert policy or solver to imitate.
            expert_name: Name of expert solver ('hgs', 'local_search').
            loss_fn: Name of loss function to use ('nll', 'kl', 'js', etc.).
            **kwargs: Arguments passed to RL4COLitModule.
        """
        # Baseline is not used in IL, but we keep the structure
        kwargs["baseline"] = "none"
        super().__init__(**kwargs)
        self.expert_policy = expert_policy
        self.expert_name = expert_name

        # Map loss functions
        self._loss_map = {
            "nll": nll_loss,
            "weighted_nll": weighted_nll_loss,
            "kl": kl_divergence_loss,
            "reverse_kl": reverse_kl_divergence_loss,
            "js": js_divergence_loss,
        }
        self.loss_fn_name = loss_fn
        self.loss_fn = self._loss_map.get(loss_fn, nll_loss)

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,
    ) -> torch.Tensor:
        """
        Compute Imitation Learning loss.

        Args:
            td: Current environment state.
            out: Policy output dictionary (not used for expert selection).
            batch_idx: Batch index.
            env: Environment instance.

        Returns:
            Scalar loss tensor.
        """
        # 1. Generate Expert Solutions
        if self.expert_policy is None:
            raise ValueError("expert_policy must be provided for ImitationLearning.")

        with torch.no_grad():
            expert_out = self.expert_policy(td, self.env)
            expert_actions = expert_out["actions"]

        # 2. Clone state and reset to ensure clean state for teacher forcing
        td_clone = safe_td_copy(td)
        td_clone = self.env.reset(td_clone)

        # 3. Teacher Forcing: Get policy log probabilities for expert actions
        teacher_out = self.policy(td_clone, self.env, actions=expert_actions)
        log_likelihood = teacher_out["log_likelihood"]

        # 4. Compute Loss using injected loss function
        if self.loss_fn_name == "weighted_nll":
            expert_reward = expert_out.get("reward", torch.ones_like(log_likelihood))
            loss = weighted_nll_loss(log_likelihood, weights=expert_reward, reduction="mean")
        elif self.loss_fn_name in ["kl", "reverse_kl", "js"]:
            # Distillation requires target distribution (logits or log_probs)
            target_log_probs = expert_out.get("log_probs", None)
            if target_log_probs is None:
                print(
                    f"[ImitationLearning] Warning: {self.loss_fn_name} requires target log_probs. Falling back to NLL."
                )
                loss = nll_loss(log_likelihood, reduction="mean")
            else:
                loss = self._loss_map[self.loss_fn_name](log_likelihood, target_log_probs, reduction="mean")
        else:
            loss = nll_loss(log_likelihood, reduction="mean")

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)

        return loss
