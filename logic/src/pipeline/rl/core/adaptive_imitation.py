"""
Adaptive Imitation Learning + RL algorithm.
Combines PPO/REINFORCE with Imitation Learning using an annealing schedule.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.reinforce import REINFORCE


class AdaptiveImitation(REINFORCE):
    """
    Combined RL + Imitation Learning with Adaptive Weight Scheduling.

    Loss = Loss_RL + lambda * Loss_IL

    Schedule:
    - lambda starts at `il_weight`
    - decays by `il_decay` each epoch
    - resets to `il_weight` if validation reward dominates patience (simulated annealing reheating)
    """

    def __init__(
        self,
        expert_policy: Any,
        il_weight: float = 1.0,
        il_decay: float = 0.95,
        patience: int = 5,
        **kwargs,
    ):
        """
        Initialize AdaptiveImitation module.

        Args:
            expert_policy: Expert policy to imitate.
            il_weight: Initial weight for imitation loss.
            il_decay: Decay factor for IL weight each epoch.
            patience: Epochs without improvement before resetting IL weight.
            **kwargs: Arguments passed to REINFORCE.
        """
        # Exclude non-serializable objects from hyperparameters
        self.save_hyperparameters(ignore=["expert_policy", "env", "policy"])
        super().__init__(**kwargs)

        # Manually remove complex objects from hparams to prevent YAML serialization errors
        if hasattr(self, "hparams"):
            keys_to_remove = []
            allowed_types = (int, float, str, bool, type(None))
            for k, v in self.hparams.items():
                # Remove known complex objects
                if k in ["expert_policy", "env", "policy"]:
                    keys_to_remove.append(k)
                    continue

                # Keep strictly primitive types
                if isinstance(v, allowed_types):
                    continue

                # Allow strictly primitive lists
                if isinstance(v, (list, tuple)):
                    if all(isinstance(x, allowed_types) for x in v):
                        continue
                    else:
                        print(f"[AdaptiveImitation] Removing non-primitive list hparam: {k}")
                        keys_to_remove.append(k)
                        continue

                # Allow strictly primitive dicts (shallow check)
                if isinstance(v, dict):
                    if all(isinstance(x, allowed_types) for x in v.values()):
                        continue
                    else:
                        print(f"[AdaptiveImitation] Removing non-primitive dict hparam: {k}")
                        keys_to_remove.append(k)
                        continue

                # Everything else
                print(f"[AdaptiveImitation] Removing complex hparam: {k} (type: {type(v)})")
                keys_to_remove.append(k)

            for k in keys_to_remove:
                self.hparams.pop(k, None)

        self.expert_policy = expert_policy
        self.il_weight = il_weight
        self.initial_il_weight = il_weight
        self.il_decay = il_decay
        self.patience = patience

        # State tracking
        self.wait = 0
        self.best_reward = float("-inf")
        self.current_il_weight = il_weight

    def calculate_loss(
        self,
        td: TensorDict,
        out: dict,
        batch_idx: int,
        env: Any = None,  # Accept env to match RL4COLitModule.shared_step
    ) -> torch.Tensor:
        """
        Compute Combined Loss: RL + IL.
        """
        # 1. RL Loss (REINFORCE)
        # Note: This computes advantage, baseline, etc.
        rl_loss = super().calculate_loss(td, out, batch_idx)

        # 2. Imitation Loss (Cross Entropy)
        # Generate expert actions
        with torch.no_grad():
            expert_out = self.expert_policy(td, self.env)
            expert_actions = expert_out["actions"]

        # Teacher Forcing: Evaluate log likelihood of expert actions under current policy
        # We need to clone td because policy/env might modify it or expert did?
        # Ideally td passed to calculate_loss is fresh from reset in shared_step
        # We use td directly to avoid RecursionError in cloning if td has cycles
        td_il = td

        il_out = self.policy(td_il, self.env, actions=expert_actions)
        log_likelihood = il_out["log_likelihood"]

        # NLL Loss
        il_loss = -log_likelihood.mean()

        # Combined Loss
        total_loss = rl_loss + self.current_il_weight * il_loss

        # Logging
        self.log("train/rl_loss", rl_loss, on_step=False, on_epoch=True)
        self.log("train/il_loss", il_loss, on_step=False, on_epoch=True)
        self.log("train/il_weight", self.current_il_weight, on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        """
        Update IL weight based on validation performance.

        Decays IL weight each epoch, or resets if patience is exceeded.
        """
        super().on_train_epoch_end()

        # Get validation reward
        # Note: validation happens *before* this hook in some Lightning versions or *after*?
        # Usually checking callback_metrics is safe for 'on_train_epoch_end' if val happens in between.
        # But standard Lightning flow: Train Epoch -> Val Epoch.
        # So at end of train epoch, we might be looking at *previous* val epoch?
        # Or we rely on the fact that we check progress.

        current_reward = self.trainer.callback_metrics.get("val/reward")

        if current_reward is not None:
            # Check for improvement
            if current_reward > self.best_reward + 1e-5:
                self.best_reward = current_reward.item()
                self.wait = 0
            else:
                self.wait += 1

            # Schedule Logic
            if self.wait >= self.patience:
                # Re-heat / Reset
                self.current_il_weight = self.initial_il_weight
                self.wait = 0
                print(f"\n[AdaptiveImitation] Re-heating IL weight to {self.initial_il_weight}")
            else:
                # Decay
                self.current_il_weight *= self.il_decay
