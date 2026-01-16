"""
Group Sequence Policy Optimization (GSPO) Implementation.

This module implements the GSPO algorithm which optimizes policy at the sequence level
using group-based advantage normalization, where the group is defined as the mini-batch.
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


class GSPOTrainer(TimeTrainer):
    """
    Group Sequence Policy Optimization (GSPO) Trainer.
    Optimizes policy at the sequence level using group-based advantage normalization.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the GSPOTrainer.
        """
        super().__init__(*args, **kwargs)
        self.epsilon = self.opts.get("gspo_epsilon", 0.2)
        self.gspo_epochs = self.opts.get("gspo_epochs", 3)
        self.mini_batch_size = self.opts.get("ppo_mini_batch_size", self.opts["batch_size"])

    def train_day(self):
        """
        Runs one day of GSPO training.
        Collection Phase -> Update Phase
        """
        self.train_day_gspo()

    def train_day_gspo(self):
        """
        GSPO Training Loop:
        1. Collect rollouts (using current policy)
        2. Compute advantages (group-based)
        3. Optimize policy using sequence-level objective

        Note: GSPO normalizes advantages within groups of samples from the same input.
        In this implementation, each batch serves as a group for normalization.
        """
        log_pi = []
        log_costs = []

        # GSPO uses sampling for exploration
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

        # Memory to store rollouts for multi-epoch updates
        rollouts = []

        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts["no_progress_bar"])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)

            if self.weight_optimizer and hasattr(self.weight_optimizer, "get_current_weights"):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)

            # Forward pass (collect data without optimization)
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(batch, batch_id, opt_step=False)

            if pi is not None:
                # Store rollout data
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
                # Infer from batch dict
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

        # GSPO Update Phase
        self.update_gspo(rollouts)

        day_duration = time.time() - start_time
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def update_gspo(self, rollouts):
        """
        Perform GSPO updates on collected rollouts with mini-batch processing.

        GSPO Key Features:
        - Group-based advantage normalization (batch-level)
        - Sequence-level importance ratio: r = exp((log π_new - log π_old) / |sequence|)
        - Clipped surrogate objective similar to PPO
        """
        if not rollouts:
            return

        # Multiple update epochs over the same data
        for _ in range(self.gspo_epochs):
            # Iterate over all collected batches
            for data in rollouts:
                batch = data["batch"]
                old_pi = data["actions"]
                old_log_probs = data["old_log_probs"]
                rewards = data["rewards"]
                values = data["values"]

                # Calculate returns and advantages
                returns = rewards
                if values is not None:
                    advantages = returns - values
                else:
                    advantages = returns

                # Group standardization of advantages (GSPO core feature)
                # Normalizes within the batch (group)
                if advantages.size(0) > 1 and advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

                batch_size = old_pi.size(0)
                indices = torch.randperm(batch_size)

                # Mini-batch loop for better gradient estimation
                for i in range(0, batch_size, self.mini_batch_size):
                    mb_idx = indices[i : i + self.mini_batch_size]

                    # Slice mini-batch data
                    mb_input = {
                        k: v[mb_idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != "edges"
                    }
                    mb_input = move_to(mb_input, self.opts["device"])

                    mb_old_pi = old_pi[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages[mb_idx]

                    # Forward pass with expert actions to get new log probabilities
                    _, new_log_probs, _, _, entropy = self.model(
                        mb_input,
                        cost_weights=self.cost_weights,
                        return_pi=False,
                        expert_pi=mb_old_pi,
                    )

                    # GSPO: Sequence-level importance ratio
                    # Standard PPO uses: ratio = exp(log_new - log_old)
                    # GSPO normalizes by sequence length: ratio = exp((log_new - log_old) / |seq|)
                    # This makes the ratio less sensitive to sequence length
                    seq_len = mb_old_pi.size(1)  # Sequence length from action tensor
                    log_ratio = new_log_probs - mb_old_log_probs
                    scaled_log_ratio = log_ratio / seq_len
                    ratio = torch.exp(scaled_log_ratio)

                    # Clipped surrogate objective (same as PPO but with GSPO ratio)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Total loss with entropy regularization
                    loss = actor_loss - self.opts.get("entropy_weight", 0.0) * entropy.mean()

                    # Optimization step
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get("max_grad_norm", 1.0))
                    self.optimizer.step()
