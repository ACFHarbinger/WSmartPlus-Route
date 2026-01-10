import time
import torch

from tqdm import tqdm
from logic.src.utils.functions import move_to
from logic.src.pipeline.reinforcement_learning.core.epoch import set_decode_type, prepare_batch
from logic.src.pipeline.reinforcement_learning.core.reinforce import TimeTrainer


class SAPOTrainer(TimeTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo_epochs = self.opts.get('ppo_epochs', 3) # Reuse PPO epochs
        self.mini_batch_size = self.opts.get('ppo_mini_batch_size', self.opts['batch_size'])
        self.tau_pos = self.opts.get('sapo_tau_pos', 0.1)
        self.tau_neg = self.opts.get('sapo_tau_neg', 1.0)
        
        # Ensure tau_neg > tau_pos as per SAPO design for stability
        if self.tau_neg <= self.tau_pos:
            print(f"WARNING: SAPO tau_neg ({self.tau_neg}) should be greater than tau_pos ({self.tau_pos}) for stability.")

    def train_day(self):
        self.train_day_sapo()

    def train_day_sapo(self):
        log_pi = []
        log_costs = []
        
        # Sampling for exploration
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
        
        rollouts = []
        
        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)
            
            if self.weight_optimizer and hasattr(self.weight_optimizer, 'get_current_weights'):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)
            
            # Forward pass (collect data)
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(batch, batch_id, opt_step=False)
            
            if pi is not None:
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
                    'batch': batch,
                    'actions': pi.detach(),
                    'old_log_probs': state_tensors['log_likelihood'].detach(),
                    'rewards': rewards, 
                    'values': values,
                    'mask': None
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
        
        # SAPO Update
        self.update_sapo(rollouts)
        
        day_duration = time.time() - start_time
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def update_sapo(self, rollouts):
        if not rollouts: 
            return

        for _ in range(self.ppo_epochs):
            for data in rollouts:
                batch = data['batch']
                old_pi = data['actions']
                old_log_probs = data['old_log_probs']
                rewards = data['rewards']
                values = data['values']
                
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
                        expert_pi=mb_old_pi,
                        imitation_mode=True
                    )
                    
                    # SAPO Loss Calculation
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    
                    # Vectorized Tau Selection
                    # tau = tau_pos if A > 0 else tau_neg
                    tau = torch.where(mb_advantages > 0, self.tau_pos, self.tau_neg)
                    
                    # Soft Gate: f(r) = (4/tau) * sigmoid(tau * (r-1))
                    # Note: We want derivative of f(r) wrt vector, which auto-diff handles if we perform operations on 'ratio'.
                    # The surrogate objective is f(r) * A.
                    
                    f_ratio = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))
                    
                    # Maximize surrogate -> Minimize negative surrogate
                    # sapo_loss = - (f_ratio * mb_advantages).mean()
                    
                    # Alternatively, verify the gradient manually:
                    # Grad L = - f'(r) * dr/dtheta * A
                    # f'(r) = 4 * sigmoid(...) * (1-sigmoid(...))
                    # This matches the blog post derivation.
                    
                    actor_loss = -(f_ratio * mb_advantages).mean()
                    
                    loss = actor_loss - self.opts.get('entropy_weight', 0.0) * entropy.mean()
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0))
                    self.optimizer.step()
