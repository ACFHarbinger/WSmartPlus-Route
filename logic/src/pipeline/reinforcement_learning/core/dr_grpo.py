"""
DR-GRPO (GRPO done Right) Implementation.

This module implements the DR-GRPO algorithm as described in recent RL literature.
It improves upon standard GRPO by using group sampling, unnormalized centered advantages,
and a sequence-level objective without length normalization.
"""
import time
import torch
from tqdm import tqdm
from logic.src.utils.functions import move_to
from logic.src.pipeline.reinforcement_learning.core.epoch import set_decode_type, prepare_batch
from logic.src.pipeline.reinforcement_learning.core.reinforce import TimeTrainer


class DRGRPOTrainer(TimeTrainer):
    """
    DR.GRPO (GRPO done Right) Trainer.
    
    Implements Group Relative Policy Optimization with specific fixes:
    1. Group sampling: G completions per input.
    2. Zero-mean centered advantages without std normalization: A = R - Mean(R_group).
    3. No sequence length normalization in the objective function.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the DRGRPOTrainer.
        """
        super().__init__(*args, **kwargs)
        self.group_size = self.opts.get('dr_grpo_group_size', 8)
        self.epsilon = self.opts.get('dr_grpo_epsilon', 0.2)
        self.dr_grpo_epochs = self.opts.get('dr_grpo_epochs', 3)
        self.mini_batch_size = self.opts.get('ppo_mini_batch_size', self.opts['batch_size'])
        
        if hasattr(self.model, 'set_decode_type'):
            self.model.set_decode_type("sampling")

    def train_day(self):
        """
        Runs one day of DR-GRPO training.
        """
        log_pi = []
        log_costs = []

        # DR-GRPO uses sampling
        set_decode_type(self.model, "sampling")

        daily_total_samples = 0
        loss_keys = list(self.cost_weights.keys()) + ['total', 'nll', 'reinforce_loss', 'baseline_loss']
        daily_loss = {key: [] for key in loss_keys}

        day_dataloader = torch.utils.data.DataLoader(
            self.baseline.wrap_dataset(self.training_dataset),
            batch_size=self.opts['batch_size'],
            pin_memory=True
        )

        start_time = time.time()
        rollouts = []

        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)

            if self.weight_optimizer and hasattr(self.weight_optimizer, 'get_current_weights'):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)

            # Group Expansion: Repeat batch inputs G times to get G samples per input
            # We assume batch items are tensors with shape [B, ...]
            # We want [B*G, ...]
            group_batch = {}
            original_batch_size = 0
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    if original_batch_size == 0:
                        original_batch_size = v.size(0)
                        
                    # Repeat each element G times: [1, 2] -> [1, 1, ..., 2, 2, ...]
                    # repeat_interleave does exactly this
                    group_batch[k] = v.repeat_interleave(self.group_size, dim=0)
                else:
                    group_batch[k] = v # Should not happen for core inputs usually, but keep as is if scalar
            
            # Forward pass to collect G samples per input
            # We use train_batch but with our expanded group_batch
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(group_batch, batch_id, opt_step=False)

            if pi is not None:
                # Rewards calculate from cost
                cost_tensor = state_tensors['cost']
                if isinstance(cost_tensor, torch.Tensor):
                    rewards = -cost_tensor.detach()
                else:
                    rewards = -torch.tensor(cost_tensor, device=self.opts['device'])

                # We don't need a value baseline network for GRPO, 
                # but we store what we have.
                # GRPO baseline is the group mean.
                
                rollouts.append({
                    'batch': group_batch, 
                    'actions': pi.detach(),
                    'old_log_probs': state_tensors['log_likelihood'].detach(),
                    'rewards': rewards,
                    'original_batch_size': original_batch_size
                })

                log_pi.append(pi.detach().cpu())

            if isinstance(batch_cost, torch.Tensor):
                log_costs.append(batch_cost.detach().cpu())
            else:
                log_costs.append(batch_cost)
            self.step += 1

            daily_total_samples += original_batch_size

            # Log metrics (using mean of the group batch)
            for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
                if key in daily_loss:
                    if isinstance(val, torch.Tensor):
                        daily_loss[key].append(val.detach().cpu().view(-1))
                    elif isinstance(val, (float, int)):
                        daily_loss[key].append(torch.tensor([val], dtype=torch.float))

        # Perform Updates
        self.update_dr_grpo(rollouts)

        day_duration = time.time() - start_time
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def update_dr_grpo(self, rollouts):
        """
        Perform DR-GRPO updates using collected rollouts.

        This method calculates the unnormalized advantages for each group and
        updates the policy using the clipped DR-GRPO objective.

        Args:
            rollouts: List of rollout dictionaries containing batch data,
                      actions, log probs, and rewards.
        """
        if not rollouts:
            return

        for _ in range(self.dr_grpo_epochs):
            for data in rollouts:
                batch = data['batch']
                old_pi = data['actions']
                old_log_probs = data['old_log_probs']
                rewards = data['rewards'] # [B*G]
                B = data['original_batch_size']
                G = self.group_size
                
                # Calculate DR-GRPO Advantages
                # Reshape rewards to [B, G]
                rewards_grouped = rewards.view(B, G)
                
                # Mean per group: [B, 1]
                group_mean = rewards_grouped.mean(dim=1, keepdim=True)
                
                # Advantage = R - Mean(R)
                # No scaling by std!
                advantages_grouped = rewards_grouped - group_mean
                
                # Flatten back to [B*G] matches batch dim
                advantages = advantages_grouped.view(-1)
                
                # Optimization Loop with Mini-Batches
                total_samples = B * G
                indices = torch.randperm(total_samples)
                
                for i in range(0, total_samples, self.mini_batch_size):
                    mb_idx = indices[i : i + self.mini_batch_size]
                    
                    # Slice mini-batch
                    mb_input = {k: v[mb_idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'edges'}
                    mb_input = move_to(mb_input, self.opts['device'])
                    
                    mb_old_pi = old_pi[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    
                    # Forward pass
                    _, new_log_probs, _, _, entropy = self.model(
                        mb_input,
                        cost_weights=self.cost_weights,
                        return_pi=False,
                        expert_pi=mb_old_pi
                    )
                    
                    # Importance Ratio
                    # Standard PPO ratio: exp(new - old)
                    # NO sequence length division (1/|o|) per Dr. GRPO paper insights
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    # Clipped Objective
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * mb_advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Total loss
                    loss = actor_loss - self.opts.get('entropy_weight', 0.0) * entropy.mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0))
                    self.optimizer.step()
