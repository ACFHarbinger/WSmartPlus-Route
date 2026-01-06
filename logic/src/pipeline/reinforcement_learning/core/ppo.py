import time
import torch
from tqdm import tqdm
from logic.src.utils.functions import move_to
from logic.src.pipeline.reinforcement_learning.core.epoch import set_decode_type, prepare_batch
from logic.src.pipeline.reinforcement_learning.core.reinforce import TimeTrainer


class PPOTrainer(TimeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo_epochs = self.opts.get('ppo_epochs', 3)
        self.eps_clip = self.opts.get('ppo_eps_clip', 0.2)
        # Default to batch_size if not specified, or use the arg
        self.mini_batch_size = self.opts.get('ppo_mini_batch_size', self.opts['batch_size'])

    def train_day(self):
        # Always use PPO collection regardless of horizon (treat day as horizon block)
        self.train_day_ppo()

    def train_day_ppo(self):
        log_pi = []
        log_costs = []
        
        # PPO uses sampling for exploration
        set_decode_type(self.model, "sampling")
        
        daily_total_samples = 0
        loss_keys = list(self.cost_weights.keys()) + ['total', 'nll', 'reinforce_loss', 'baseline_loss', 'imitation_loss']
        daily_loss = {key: [] for key in loss_keys}
        
        day_dataloader = torch.utils.data.DataLoader(
            self.baseline.wrap_dataset(self.training_dataset), 
            batch_size=self.opts['batch_size'], 
            pin_memory=True
        )
        
        start_time = time.time()
        
        # Memory to store trajectories
        # List of (batch_input, actions, old_log_probs, rewards, values, mask)
        rollouts = []
        
        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)
            
            if self.weight_optimizer and hasattr(self.weight_optimizer, 'get_current_weights'):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)
            
            # Forward pass (collect data)
            # We use opt_step=False because we verify gradients later in PPO loop
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(batch, batch_id, opt_step=False)
            
            if pi is not None:
                # Store rollout data
                # state_tensors['log_likelihood'] is old_log_probs
                # state_tensors['bl_val'] is value
                
                cost_tensor = state_tensors['cost']
                if isinstance(cost_tensor, torch.Tensor):
                    rewards = -cost_tensor.detach()
                else:
                     rewards = -torch.tensor(cost_tensor, device=self.opts['device'])

                bl_val = state_tensors['bl_val']
                if bl_val is not None and isinstance(bl_val, torch.Tensor):
                    values = bl_val.detach()
                else:
                    values = bl_val
                
                rollouts.append({
                    'batch': batch, # Original input
                    'actions': pi.detach(),
                    'old_log_probs': state_tensors['log_likelihood'].detach(),
                    'rewards': rewards, 
                    'values': values,
                    'mask': None # Mask is usually inferred or part of batch
                })
                
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
                    current_batch_size = self.opts['batch_size']
            
            daily_total_samples += current_batch_size
            
            for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
                if key in daily_loss:
                    if isinstance(val, torch.Tensor): 
                        daily_loss[key].append(val.detach().cpu().view(-1))
                    elif isinstance(val, (float, int)):
                        daily_loss[key].append(torch.tensor([val], dtype=torch.float))
        
        # PPO Update
        self.update_ppo(rollouts)
        
        day_duration = time.time() - start_time
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def update_ppo(self, rollouts):
        if not rollouts: 
            return

        # Flatten rollouts if needed, but here we process list of batches
        # We perform PPO epochs
        
        for _ in range(self.ppo_epochs):
            # Iterate over all collected batches
            for data in rollouts:
                # Retrieve data
                batch = data['batch']
                old_pi = data['actions']
                old_log_probs = data['old_log_probs']
                rewards = data['rewards'] # shape [B]
                values = data['values'] # shape [B]
                
                # Calculate Advantage
                # For basic PPO without GAE-Lambda across time steps (since VRP is often 1-step MDP or we treat partial tour construction differently),
                # If these are complete tours:
                # Advantage = Reward - Value
                # Returns = Reward
                
                returns = rewards
                if values is not None:
                    advantages = returns - values
                else:
                    advantages = returns
                
                # Normalize advantages
                if advantages.size(0) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Re-evaluate policies
                # We need to call model forward pass forcing the actions 'old_pi'
                
                batch_size = old_pi.size(0)
                # Use CPU indices for safe slicing of potential CPU batch
                indices = torch.randperm(batch_size)
                
                # Mini-batch loop
                for i in range(0, batch_size, self.mini_batch_size):
                    mb_idx = indices[i : i + self.mini_batch_size]
                    
                    # Slice data
                    mb_input = {k: v[mb_idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'edges'} 
                    mb_input = move_to(mb_input, self.opts['device'])
                    
                    mb_old_pi = old_pi[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_returns = returns[mb_idx]
                    
                    # Forward pass with expert_pi to get new log_probs and entropy
                    # We pass kl_loss=True to 'forward' if we want KL, but for PPO we just need probabilities.
                    # 'expert_pi' arg forces selection.
                    
                    # Model forward signature: (input, cost_weights, return_pi, pad, mask, expert_pi, **kwargs)
                    
                    _, new_log_probs, _, _, entropy = self.model(
                        mb_input, 
                        cost_weights=self.cost_weights, 
                        return_pi=False, 
                        expert_pi=mb_old_pi,
                        imitation_mode=True # To enable pure evaluation mode
                    )
                    # Note: AttentionModel.forward with expert_pi returns log_probs of those actions.
                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss (if we have a value head/critic)
                    # Currently we don't have explicit value head update here unless 'bl_loss' logic is added.
                    # Standard PPO has value loss.
                    # If model has value head, we should update it.
                    # If baseline is separate (CriticNetwork), we update it separately?
                    # The Baseline object handles its own update usually (if it's simple baseline).
                    # If we use CriticNetwork baseline, it has 'eval' and 'train'.
                    # Here we are inside 'train_day', usually baseline is updated via 'baseline.eval' providing loss?
                    # No, usually baseline is updated after step or inside batch.
                    
                    # For now, sticking to actor loss optimization as primary.
                    # Ideally add value loss if model outputs value.
                    
                    loss = actor_loss - self.opts.get('entropy_weight', 0.0) * entropy.mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0))
                    self.optimizer.step()
