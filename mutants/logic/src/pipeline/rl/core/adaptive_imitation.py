"""
Adaptive Imitation Learning + RL algorithm.
Combines PPO/REINFORCE with Imitation Learning using an annealing schedule.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from logic.src.pipeline.rl.core.losses import (
    kl_divergence_loss,
    nll_loss,
    reverse_kl_divergence_loss,
    weighted_nll_loss,
)
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
        threshold: float = 0.05,
        decay_step: int = 1,
        epsilon: float = 1e-5,
        loss_fn: str = "weighted_nll",
        **kwargs,
    ):
        """
        Initialize AdaptiveImitation module.

        Args:
            expert_policy: Expert policy to imitate.
            il_weight: Initial weight for imitation loss.
            il_decay: Decay factor for IL weight each epoch.
            patience: Epochs without improvement before resetting IL weight.
            threshold: Threshold for reannealing.
            decay_step: Number of epochs to decay IL weight.
            loss_fn: Name of loss function to use ('weighted_nll', 'nll', etc.).
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
        self.threshold = threshold
        self.decay_step = decay_step
        self.epsilon = epsilon
        self.stop_il = False

        # Map loss functions
        self._loss_map = {
            "nll": nll_loss,
            "weighted_nll": weighted_nll_loss,
            "kl": kl_divergence_loss,
            "reverse_kl": reverse_kl_divergence_loss,
        }
        self.loss_fn_name = loss_fn
        self.loss_fn = self._loss_map.get(loss_fn, weighted_nll_loss)

        # State tracking
        self.wait = 0
        self.best_reward = float("-inf")
        self.current_il_weight = il_weight

    def _sanitize_td(self, td: TensorDict) -> TensorDict:
        """
        Sanitize TensorDict to remove potential recursive references or unnecessary nesting.

        This constructs a new TensorDict containing only the leaf Tensors from the original,
        breaking any reference cycles that might cause RecursionError during cloning or printing.
        """
        # Iterate over all keys in the TensorDict and keep only Tensors
        # to ensure we break any nesting or recursive structures.
        safe_data = {}
        for key in td.keys():
            val = td.get(key)
            if isinstance(val, torch.Tensor):
                safe_data[key] = val

        return TensorDict(safe_data, batch_size=td.batch_size, device=td.device)

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
        rl_loss = super().calculate_loss(td, out, batch_idx, env=env)

        # 2. Imitation Loss (Cross Entropy)
        # Generate expert actions
        if not self.stop_il:
            with torch.no_grad():
                expert_out = self.expert_policy(td, self.env)
                expert_actions = expert_out["actions"]
                expert_reward = expert_out.get("reward", None)
        else:
            expert_reward = None
            expert_actions = None
            expert_reward = None
            expert_cost = None
            expert_out = None

        # Conditional Imitation: Check if expert provides significant improvement
        # We compare means to decide whether to run the expensive forward pass to save compute
        do_imitation = not self.stop_il
        current_reward = out.get("reward", None)
        if do_imitation and expert_reward is not None and current_reward is not None:
            if expert_reward.max() <= current_reward.min() + self.threshold:
                do_imitation = False
                print("\n[AdaptiveImitation] Stopping IL for this batch")

        if do_imitation:
            td_il = self.env.reset(td)

            il_out = self.policy(td_il, self.env, actions=expert_actions)
            log_likelihood = il_out["log_likelihood"]

            # Compute modular loss
            if self.loss_fn_name == "weighted_nll":
                il_loss = weighted_nll_loss(
                    log_likelihood,
                    weights=(expert_reward - current_reward),
                    reduction="mean",
                )
            elif self.loss_fn_name == "nll":
                il_loss = nll_loss(log_likelihood, reduction="mean")
            elif self.loss_fn_name in ["kl", "reverse_kl", "js"]:
                target_log_probs = expert_out.get("log_probs", None)
                if target_log_probs is None:
                    print(
                        f"[AdaptiveImitation] Warning: {self.loss_fn_name} requires target log_probs. Falling back to weighted_nll."
                    )
                    il_loss = weighted_nll_loss(log_likelihood, (expert_reward - current_reward), reduction="mean")
                else:
                    il_loss = self._loss_map[self.loss_fn_name](log_likelihood, target_log_probs, reduction="mean")
            else:
                il_loss = weighted_nll_loss(log_likelihood, (expert_reward - current_reward), reduction="mean")
        else:
            il_loss = torch.tensor(0.0, device=self.device)

        # Combined Loss
        total_loss = rl_loss + self.current_il_weight * il_loss

        # Logging
        self.log("train/rl_loss", rl_loss, on_step=False, on_epoch=True)
        self.log("train/il_loss", il_loss, on_step=False, on_epoch=True)
        self.log("train/il_weight", self.current_il_weight, on_step=False, on_epoch=True)

        if expert_reward is not None:
            self.log("train/expert_reward", expert_reward.mean(), on_step=False, on_epoch=True)
        expert_cost = expert_out.get("cost", None)
        if expert_cost is not None:
            self.log("train/expert_cost", expert_cost.mean(), on_step=False, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        """
        Update IL weight based on validation performance.

        Decays IL weight each epoch, or resets if patience is exceeded.
        """
        super().on_train_epoch_end()

        current_reward = self.trainer.callback_metrics.get("val/reward")
        if self.current_il_weight <= self.threshold:
            self.stop_il = True
        if current_reward is not None and not self.stop_il:
            # Check for improvement
            if current_reward > self.best_reward + self.epsilon:
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
                print(f"\n[AdaptiveImitation] IL weight decayed to {self.current_il_weight}")
