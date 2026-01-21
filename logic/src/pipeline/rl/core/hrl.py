"""
Hierarchical Reinforcement Learning (HRL) module.

Implements a Manager-Worker architecture:
- Manager: GATLSTManager (decides if/what to collect)
- Worker: ConstructivePolicy (decides the route)
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.envs.base import RL4COEnvBase
from logic.src.models.gat_lstm_manager import GATLSTManager
from logic.src.models.policies.base import ConstructivePolicy


class HRLModule(pl.LightningModule):
    """
    Lightning module for Hierarchical RL.

    Coordinates a high-level manager and a low-level worker.
    """

    def __init__(
        self,
        manager: GATLSTManager,
        worker: ConstructivePolicy,
        env: RL4COEnvBase,
        lr: float = 1e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        lambda_mask_aux: float = 0.0,
        entropy_coef: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["manager", "worker", "env"])
        self.manager = manager
        self.worker = worker
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.lambda_mask_aux = lambda_mask_aux
        self.entropy_coef = entropy_coef

        # Disable automatic optimization for PPO multiple updates
        self.automatic_optimization = False

    def training_step(self, batch: TensorDict, batch_idx: int):
        """
        Combined training step for Manager and Worker with PPO.
        """
        opt = self.optimizers()

        # --- 1. Collection Phase ---
        td = self.env.reset(batch)

        # Prepare inputs for manager
        static = td["locs"]
        if "demand_history" in td.keys():
            dynamic = td["demand_history"]
        else:
            dynamic = td["demand"].unsqueeze(-1).expand(-1, -1, 10)

        global_features = torch.stack(
            [
                td["demand"].mean(dim=1),
                td["demand"].max(dim=1)[0],
            ],
            dim=-1,
        )

        # Manager Decision (Stores states/actions in manager memory)
        # Note: deterministice=False by default in Select Action
        mask_action, gate_action, manager_value = self.manager.select_action(
            static, dynamic, global_features, deterministic=False
        )

        # Worker Decision (Routing)
        dispatch_indices = (gate_action == 1).nonzero().squeeze(-1)
        total_reward = torch.zeros(td.batch_size, device=td.device)

        if len(dispatch_indices) > 0:
            td_worker = td[dispatch_indices]
            # Apply manager's mask_action to td_worker['visited']
            td_worker["visited"] = td_worker["visited"] | (mask_action[dispatch_indices] == 0)
            out_worker = self.worker(td_worker, self.env)
            total_reward[dispatch_indices] = out_worker["reward"]

        # Store rewards in manager buffer (assuming 1-step horizon for now)
        self.manager.rewards.append(total_reward)

        # Log metrics
        self.log("train/manager_gate_rate", gate_action.float().mean())
        self.log("train/manager_visit_rate", mask_action.float().mean())
        self.log("train/reward", total_reward.mean(), prog_bar=True)

        # --- 2. PPO Optimization Phase ---
        # Replicates logic from manager_train.py::update_manager_ppo

        # Prepare Tensors from Buffer
        # For now we handle single-step (T=1), but structure allows accumulation if we loops
        T = len(self.manager.rewards)
        B = self.manager.rewards[0].size(0)

        rewards_tensor = torch.stack(self.manager.rewards)  # (T, B)
        # Compute Returns (G_t)
        returns_tensor = torch.zeros_like(rewards_tensor)
        running_return = torch.zeros(B, device=self.device)
        for t in reversed(range(T)):
            running_return = rewards_tensor[t] + self.gamma * running_return
            returns_tensor[t] = running_return

        returns = returns_tensor.flatten()  # (T*B)

        # Concatenate memory buffers
        # Assuming select_action populated these
        old_states_static = torch.cat(self.manager.states_static)
        old_states_dynamic = torch.cat(self.manager.states_dynamic)
        old_states_global = torch.cat(self.manager.states_global)
        old_mask_actions = torch.cat(self.manager.actions_mask)
        old_gate_actions = torch.cat(self.manager.actions_gate)
        old_log_probs_mask = torch.cat(self.manager.log_probs_mask)
        old_log_probs_gate = torch.cat(self.manager.log_probs_gate)
        old_values = torch.cat(self.manager.values).squeeze(-1)
        old_target_masks = torch.cat(self.manager.target_masks) if self.manager.target_masks else None

        # Calculate Advantages
        advantages = returns - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # PPO Inner Loops
        old_states_static.size(0)
        mean_loss = 0

        # HICRA Credit Assignment weights
        # Higher weight for instances with critical nodes (overflow > 0.9)
        b_overflow = (old_states_dynamic[:, :, -1] > 0.9).float().sum(dim=1)
        credit_weight = 1.0 + (b_overflow * 0.5)
        credit_weight = credit_weight / (credit_weight.mean() + 1e-8)

        for _ in range(self.ppo_epochs):
            # Full batch processing or mini-batching?
            # manager_train.py used mini-batches. We'll use full batch if small enough, or simple batching.
            # Here we just use the full collected batch (since batch_size of Lightning is usually GPU-sized)

            # Forward pass for current policy
            mask_logits, gate_logits, values = self.manager(old_states_static, old_states_dynamic, old_states_global)
            values = values.squeeze(-1)

            # New Log Probs
            mask_dist = torch.distributions.Categorical(logits=mask_logits)
            gate_dist = torch.distributions.Categorical(logits=gate_logits)

            new_log_probs_mask = mask_dist.log_prob(old_mask_actions).sum(dim=1)
            new_log_probs_gate = gate_dist.log_prob(old_gate_actions)

            new_log_probs = new_log_probs_mask + new_log_probs_gate
            old_log_probs = old_log_probs_mask + old_log_probs_gate

            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate Loss with Credit Assignment
            surr1 = ratio * advantages * credit_weight
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages * credit_weight
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy = mask_dist.entropy().mean() + gate_dist.entropy().mean()

            loss = actor_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # Aux Loss (if target mask available)
            if self.lambda_mask_aux > 0 and old_target_masks is not None:
                logits_diff = mask_logits[:, :, 1] - mask_logits[:, :, 0]
                pos_weight = torch.tensor([50.0], device=self.device)
                loss_mask_aux = F.binary_cross_entropy_with_logits(logits_diff, old_target_masks, pos_weight=pos_weight)
                loss += self.lambda_mask_aux * loss_mask_aux

            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt.step()

            mean_loss += loss.item()

        # Clear memory after update
        self.manager.clear_memory()

        self.log("train/manager_loss", mean_loss / self.ppo_epochs)

    def configure_optimizers(self):
        # We only train Manager with this PPO logic usually?
        # worker is usually pre-trained or trained separately?
        # In manager_train.py, it updated MANAGER.
        # If we want to train worker too, it should be in the loop or separate.
        # Here we follow manager_train.py: train manager.
        return torch.optim.Adam(self.manager.parameters(), lr=self.lr)
