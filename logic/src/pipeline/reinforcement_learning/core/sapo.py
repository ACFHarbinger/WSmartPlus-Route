"""
Soft Adaptive Policy Optimization (SAPO) Implementation.

This module implements SAPO, which adaptively adjusts clipping based on the sign
of the advantage, using soft gates instead of hard clipping for smoother updates.
"""

import time

import torch
from tqdm import tqdm

from logic.src.pipeline.reinforcement_learning.core.epoch import (
    prepare_batch,
    set_decode_type,
)
from logic.src.pipeline.reinforcement_learning.core.reinforce import TimeTrainer
from logic.src.utils.functions import move_to


class SAPOTrainer(TimeTrainer):
    """
    Soft Adaptive Policy Optimization (SAPO) Trainer.

    SAPO adaptively adjusts clipping based on advantage sign:
    - Uses soft gates instead of hard clipping: f(r) = (4/tau) * sigmoid(tau * (r-1))
    - tau_pos for positive advantages (more aggressive updates)
    - tau_neg for negative advantages (more conservative updates)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the SAPOTrainer.
        """
        super().__init__(*args, **kwargs)
        self.ppo_epochs = self.opts.get("ppo_epochs", 3)
        self.mini_batch_size = self.opts.get("ppo_mini_batch_size", self.opts["batch_size"])
        self.tau_pos = self.opts.get("sapo_tau_pos", 0.1)
        self.tau_neg = self.opts.get("sapo_tau_neg", 1.0)

    def train_day(self):
        """
        Execute training for a single day using SAPO logic.
        """
        self.train_day_sapo()

    def train_day_sapo(self):
        """
        Collect trajectories and update policy using SAPO.
        """
        log_pi = []
        log_costs = []

        # Sampling for exploration
        set_decode_type(self.model, "sampling")

        daily_total_samples = 0
        loss_keys = list(self.cost_weights.keys()) + [
            "total",
            "nll",
            "reinforce_loss",
            "baseline_loss",
            "imitation_loss",
        ]
        daily_loss = {key: [] for key in loss_keys}

        day_dataloader = torch.utils.data.DataLoader(
            self.baseline.wrap_dataset(self.training_dataset),
            batch_size=self.opts["batch_size"],
            pin_memory=True,
        )

        start_time = time.time()

        rollouts = []

        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts["no_progress_bar"])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)

            if self.weight_optimizer and hasattr(self.weight_optimizer, "get_current_weights"):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)

            # Forward pass (collect data)
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(batch, batch_id, opt_step=False)

            if pi is not None:
                cost_tensor = state_tensors["cost"]
                if isinstance(cost_tensor, torch.Tensor):
                    rewards = -cost_tensor.detach()
                else:
                    rewards = -torch.tensor(cost_tensor, device=self.opts["device"])

                bl_val = state_tensors["bl_val"]
                if bl_val is not None and isinstance(bl_val, torch.Tensor):
                    values = bl_val.detach()
                else:
                    values = bl_val

                rollouts.append(
                    {
                        "batch": batch,
                        "actions": pi.detach(),
                        "old_log_probs": state_tensors["log_likelihood"].detach(),
                        "rewards": rewards,
                        "values": values,
                        "mask": None,
                    }
                )

                log_pi.append(pi.detach().cpu())

            if isinstance(batch_cost, torch.Tensor):
                log_costs.append(batch_cost.detach().cpu())
            else:
                log_costs.append(batch_cost)
            self.step += 1
            if pi is not None:
                current_batch_size = pi.size(0)
            else:
                first_val = next(iter(batch.values()))
                if isinstance(first_val, torch.Tensor):
                    current_batch_size = first_val.size(0)
                else:
                    current_batch_size = self.opts["batch_size"]

            daily_total_samples += current_batch_size

            for key, val in zip(
                list(c_dict.keys()) + list(l_dict.keys()),
                list(c_dict.values()) + list(l_dict.values()),
            ):
                if key in daily_loss:
                    if isinstance(val, torch.Tensor):
                        daily_loss[key].append(val.detach().cpu().view(-1))
                    elif isinstance(val, (float, int)):
                        daily_loss[key].append(torch.tensor([val], dtype=torch.float))

        # SAPO Update
        self.update_sapo(rollouts)

        day_duration = time.time() - start_time
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def update_sapo(self, rollouts):
        """
        Perform SAPO updates on collected rollouts.

        Calculates advantages and applies the Soft Adaptive Policy Optimization update,
        using an asymmetric soft-clipping mechanism based on advantage sign.

        Args:
            rollouts: List of rollout dictionaries.
        """
        if not rollouts:
            return

        for _ in range(self.ppo_epochs):
            for data in rollouts:
                batch = data["batch"]
                old_pi = data["actions"]
                old_log_probs = data["old_log_probs"]
                rewards = data["rewards"]
                values = data["values"]

                returns = rewards
                if values is not None:
                    advantages = returns - values
                else:
                    advantages = returns

                if advantages.size(0) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                batch_size = old_pi.size(0)
                indices = torch.randperm(batch_size)

                for i in range(0, batch_size, self.mini_batch_size):
                    mb_idx = indices[i : i + self.mini_batch_size]

                    mb_input = {
                        k: v[mb_idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "edges"
                    }
                    mb_input = move_to(mb_input, self.opts["device"])

                    mb_old_pi = old_pi[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages[mb_idx]

                    # Forward pass
                    _, new_log_probs, _, _, entropy = self.model(
                        mb_input,
                        cost_weights=self.cost_weights,
                        return_pi=False,
                        expert_pi=mb_old_pi,
                    )

                    # SAPO Loss Calculation
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    # Adaptive tau selection based on advantage sign
                    # tau_pos (smaller): More aggressive updates for good actions (positive advantages)
                    # tau_neg (larger): More conservative updates for bad actions (negative advantages)
                    tau = torch.where(mb_advantages > 0, self.tau_pos, self.tau_neg)

                    # Soft gate function replaces hard clipping in PPO
                    # f(r) = (4/tau) * sigmoid(tau * (r-1))
                    # Properties:
                    # - Smooth and differentiable (unlike PPO's min operation)
                    # - f(1) = 2 (neutral point where ratio = 1)
                    # - Smaller tau -> steeper sigmoid -> more aggressive clipping
                    # - Larger tau -> gentler sigmoid -> more conservative clipping
                    f_ratio = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))

                    # SAPO objective: E[f(r) * A]
                    # Maximize expected advantage-weighted soft-gated ratio
                    actor_loss = -(f_ratio * mb_advantages).mean()

                    loss = actor_loss - self.opts.get("entropy_weight", 0.0) * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get("max_grad_norm", 1.0))
                    self.optimizer.step()
