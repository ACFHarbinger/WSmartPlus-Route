"""
REINFORCE Trainer Implementations.

This module provides concrete implementations of REINFORCE-based trainers for various
training scenarios. All trainers inherit from BaseReinforceTrainer and implement
specific training strategies:

- StandardTrainer: Basic REINFORCE with optional imitation learning
- TimeTrainer: Sequential time-dependent training with environment state evolution
- RWATrainer: Reward Weight Adaptation using learnable meta-models
- ContextualBanditTrainer: Dynamic weight selection using contextual bandits
- TDLTrainer: Task-Dependent Learning with adaptive cost weights
- MORLTrainer: Multi-Objective Reinforcement Learning
- HyperNetworkTrainer: Meta-learning with hypernetworks
- HRLTrainer: Hierarchical Reinforcement Learning with manager-worker architecture

Each trainer implements the train_day() method to define its specific training logic
while inheriting the common training loop structure from BaseReinforceTrainer.
"""

import time
import torch
import numpy as np
from tqdm import tqdm

from logic.src.utils.functions import move_to
from logic.src.pipeline.reinforcement_learning.core.epoch import (
    prepare_batch, prepare_time_dataset, update_time_dataset, set_decode_type
)
from logic.src.pipeline.reinforcement_learning.core.post_processing import local_search_2opt_vectorized
from logic.src.pipeline.reinforcement_learning.meta import (
    RewardWeightOptimizer, WeightContextualBandit,
    CostWeightManager, MORLWeightOptimizer
)
from logic.src.pipeline.reinforcement_learning.policies.hgs_vectorized import VectorizedHGS
from logic.src.models import WeightAdjustmentRNN, HypernetworkOptimizer, GATLSTManager
from logic.src.pipeline.reinforcement_learning.core.base import BaseReinforceTrainer


class StandardTrainer(BaseReinforceTrainer):
    """
    Standard REINFORCE trainer with optional imitation learning.

    Implements vanilla policy gradient training with:
    - REINFORCE algorithm for policy optimization
    - Baseline for variance reduction (exponential, critic, rollout, or POMO)
    - Optional imitation learning from expert solutions (2-opt or HGS)
    - Gradient accumulation support
    - Mixed precision training via GradScaler

    The train_batch method implements the core REINFORCE update:
    1. Sample action from policy: a ~ π(·|s)
    2. Compute reward/cost: R = cost(a, s)
    3. Calculate advantage: A = R - baseline(s)
    4. Policy gradient loss: L = A * log π(a|s)
    5. Optional imitation loss: L_imit = -log π(a_expert|s)
    6. Total loss: L_total = L + λ * L_imit

    This trainer is used for standard (non-temporal) problems or as a base
    for more advanced training strategies.
    """
    def initialize_training_dataset(self):
        """
        Initialize the training dataset (no-op for standard trainer).
        """
        pass

    def train_day(self):
        """
        Execute training for a single day (iterate over dataloader).
        Equivalent to _train_single_day in reinforce.py.
        """
        log_pi = []
        log_costs = []
        
        # Set decode type to sampling for training
        set_decode_type(self.model, "sampling")
        
        daily_total_samples = 0
        loss_keys = list(self.cost_weights.keys()) + ['total', 'nll', 'reinforce_loss']
        if self.opts['baseline'] is not None:
            loss_keys.append('baseline_loss')
        if self.opts.get('imitation_weight', 0) > 0:
            loss_keys.append('imitation_loss')
            loss_keys.append('expert_cost')

        daily_loss = {key: [] for key in loss_keys}  
        
        day_dataloader = torch.utils.data.DataLoader(
            self.baseline.wrap_dataset(self.training_dataset), 
            batch_size=self.opts['batch_size'], 
            pin_memory=True
        )
        
        start_time = time.time()
        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)
            
            # Per-batch weight update if optimizer supports it
            if self.weight_optimizer and hasattr(self.weight_optimizer, 'get_current_weights'):
                current_weights = self.weight_optimizer.get_current_weights()
                self.cost_weights.update(current_weights)
            
            pi, c_dict, l_dict, batch_cost, _ = self.train_batch(batch, batch_id)
            
            if pi is not None:
                log_pi.append(pi.detach().cpu())
            log_costs.append(batch_cost.detach().cpu())
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
        
        day_duration = time.time() - start_time
        
        # Store for post-processing
        self.daily_loss = daily_loss
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def train_batch(self, batch, batch_id, opt_step=True):
        """
        Train on a single batch of data.

        Args:
            batch: Batch of data tokens.
            batch_id (int): Identifier for the batch.
            opt_step (bool): Whether to perform an optimizer step (backprop).

        Returns:
            Tuple: (pi, cost_dict, loss_dict, cost_mean, state_tensors)
        """
        # Logic extracted from train_batch_reinforce
        x, bl_val = self.baseline.unwrap_batch(batch)
        x = move_to(x, self.opts['device'])
        bl_val = move_to(bl_val, self.opts['device']) if bl_val is not None else None
        
        if self.scaler is not None:
            autocast_context = torch.cuda.amp.autocast(dtype=torch.float16) 
            autocast_context.__enter__()

        try:
            mask = batch.get('hrl_mask', None)
            if mask is not None:
                mask = move_to(mask, self.opts['device'])
            
            cost, log_likelihood, c_dict, pi, entropy = self.model(
                x, cost_weights=self.cost_weights, 
                return_pi=self.opts['train_time'], 
                pad=self.opts['train_time'], 
                mask=mask
            )

            if self.opts.get('pomo_size', 0) > 1:
                cost_pomo = cost.view(-1, self.opts['pomo_size'])
                bl_val = cost_pomo.mean(dim=1, keepdim=True).expand_as(cost_pomo).reshape(-1)
                bl_loss = torch.tensor([0.0], device=self.opts['device'])
            else:
                bl_val, bl_loss = self.baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
                if not isinstance(bl_loss, torch.Tensor):
                    bl_loss = torch.tensor([bl_loss], device=self.opts['device'], dtype=torch.float)
                elif bl_loss.dim() == 0:
                    bl_loss = bl_loss.unsqueeze(0)

            reinforce_loss = (cost - bl_val) * log_likelihood
            entropy_loss = -self.opts.get('entropy_weight', 0.0) * entropy if entropy is not None else 0.0
            
            imitation_loss = torch.tensor(0.0, device=self.opts['device'])
            curr_imitation_weight = self.opts.get('imitation_weight', 0.0) # Decay handled externally
            
            expert_cost_tensor = None
            if curr_imitation_weight >= self.opts['lr_model'] and pi is not None:
                dist_matrix = x.get('dist', None)
                expert_pi = None
                
                if self.opts.get('imitation_mode', '2opt') == 'hgs':
                    # HGS requires demands and capacity
                    # Assuming x['waste'] is normalized demand and capacity is 1.0 (typical for RL)
                    # or recover from opts
                    demands = x.get('waste', None)
                    # Flatten demands if needed? x['waste'] is usually (B, N) or (B, N, 1) or (B, 1, N)
                    if demands is not None:
                        if demands.dim() == 3: 
                            demands = demands.squeeze(1).squeeze(1)
                        if demands.dim() == 2 and demands.size(1) == 1: 
                            demands = demands.squeeze(1)
                        
                        # Pad with depot demand (0) if size matches graph_size (customers only)
                        # Assuming dist_matrix is (B, N+1, N+1) and demands is (B, N)
                        if demands.size(1) == dist_matrix.size(1) - 1:
                            demands = torch.cat([torch.zeros((demands.size(0), 1), device=demands.device), demands], dim=1)
                        
                    vehicle_capacity = 1.0 # Default for normalized envs
                    
                    if demands is not None and dist_matrix is not None:
                        # Expand dist_matrix if shared (B=1)
                        if dist_matrix.size(0) == 1 and pi.size(0) > 1:
                            dist_matrix = dist_matrix.expand(pi.size(0), -1, -1)
                        
                        # Prepare Giant Tours from pi (Constructive solution with 0s)
                        # Remove 0s to get permutation
                        pi_giant_list = []
                        valid_hgs_indices = []
                        expected_len = dist_matrix.size(1) - 1 # N nodes
                        
                        pi_cpu = pi.detach().cpu().numpy()
                        for i in range(pi.size(0)):
                            # Filter 0s
                            tour = pi_cpu[i]
                            giant = tour[tour != 0]
                            # Check if valid permutation (length check is a good proxy for now)
                            # If partial tour, complete with random unvisited nodes for HGS
                            if len(giant) < expected_len:
                                # Find nodes in range [1, expected_len] not in giant
                                all_nodes = np.arange(1, expected_len + 1)
                                missing = np.setdiff1d(all_nodes, giant)
                                if len(missing) > 0:
                                    np.random.shuffle(missing)
                                    giant = np.concatenate([giant, missing])
                            
                            if len(giant) == expected_len:
                                pi_giant_list.append(giant)
                                valid_hgs_indices.append(i)
                            else:
                                pass # Discard if still invalid somehow

                        if pi_giant_list:
                            # Convert list of numpy arrays to a single numpy array first
                            giant_tours_np = torch.from_numpy(np.array(pi_giant_list)).to(self.opts['device'])
                            
                            hgs_dist = dist_matrix
                            if dist_matrix.size(0) == pi.size(0):
                                hgs_dist = dist_matrix[valid_hgs_indices]
                                
                            hgs_demands = demands
                            if demands.size(0) == pi.size(0):
                                hgs_demands = demands[valid_hgs_indices]

                            hgs_solver = VectorizedHGS(hgs_dist, hgs_demands, vehicle_capacity, device=self.opts['device'])
                            try:
                                expert_pi_valid, _ = hgs_solver.solve(giant_tours_np)
                                # Dynamic padding for expert_pi based on HGS output length
                                max_hgs_len = max(len(r) for r in expert_pi_valid) if expert_pi_valid else 0
                                if max_hgs_len > 0:
                                    expert_pi = torch.zeros((pi.size(0), max_hgs_len), dtype=torch.long, device=self.opts['device'])
                                    for idx, batch_idx in enumerate(valid_hgs_indices):
                                        row = expert_pi_valid[idx]
                                        expert_pi[batch_idx, :len(row)] = torch.tensor(row, device=self.opts['device'])
                            except Exception:
                                pass
                                    
                        if expert_pi is not None and expert_pi.size(1) > 0 and expert_pi[0, 0] == 0:
                            expert_pi = expert_pi[:, 1:]

                elif self.opts.get('imitation_mode', '2opt') == '2opt':
                    if dist_matrix is not None:
                        if dist_matrix.size(0) == 1 and pi.size(0) > 1:
                            dist_matrix = dist_matrix.expand(pi.size(0), -1, -1)
                        with torch.no_grad():
                            pi_opt_with_depot = local_search_2opt_vectorized(pi, dist_matrix, self.opts['two_opt_max_iter'])
                            expert_pi = pi_opt_with_depot[:, 1:]
                                    
                if expert_pi is not None:
                    # Use robust MLE (kl_loss=False + clamping in decoder) for imitation
                    _, expert_log_likelihood, _, _, _ = self.model(
                        x, cost_weights=self.cost_weights, 
                        return_pi=False, 
                        pad=self.opts['train_time'],
                        expert_pi=expert_pi
                    )
                    imitation_loss = -expert_log_likelihood.mean()
                    
                    # Compute Expert Cost for Reannealing
                    with torch.no_grad():
                        expert_cost, _, _ = self.problem.get_costs(x, expert_pi, self.cost_weights, dist_matrix)
                        expert_cost_tensor = expert_cost.mean()

            
            loss = reinforce_loss.mean() + bl_loss.mean() + entropy_loss.mean() + curr_imitation_weight * imitation_loss
            loss = loss / self.opts.get('accumulation_steps', 1)

            if opt_step:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if (batch_id + 1) % self.opts.get('accumulation_steps', 1) == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['max_grad_norm'])
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_id + 1) % self.opts.get('accumulation_steps', 1) == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['max_grad_norm'])
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            
            l_dict = {
                'total': loss.item() * self.opts.get('accumulation_steps', 1),
                'reinforce_loss': reinforce_loss.mean().item(),
                'baseline_loss': bl_loss.mean().item() if isinstance(bl_loss, torch.Tensor) else bl_loss,
                'nll': -log_likelihood.mean().item() 
            }
            if curr_imitation_weight > 0:
                l_dict['imitation_loss'] = imitation_loss.item() if isinstance(imitation_loss, torch.Tensor) else imitation_loss
            
            if expert_cost_tensor is not None:
                l_dict['expert_cost'] = expert_cost_tensor.item() if isinstance(expert_cost_tensor, torch.Tensor) else expert_cost_tensor

            state_tensors = None
            if not opt_step:
                state_tensors = {
                    'log_likelihood': log_likelihood,
                    'cost': cost,
                    'bl_val': bl_val,
                    'entropy': entropy,
                    'imitation_loss': imitation_loss,
                    'curr_imitation_weight': curr_imitation_weight,
                    'pi': pi # Added pi for off-policy algorithms
                }

            if self.scaler is not None and opt_step:
                autocast_context.__exit__(None, None, None)
            
            if not opt_step and self.scaler is not None:
                autocast_context.__exit__(None, None, None)

            return pi, c_dict, l_dict, cost.mean(), state_tensors
            
        except Exception:
            if self.scaler is not None: 
                autocast_context.__exit__(None, None, None)
            raise


class TimeTrainer(StandardTrainer):
    """
    Trainer for time-dependent Reinforcement Learning with environment state evolution.

    Extends StandardTrainer to handle sequential waste collection scenarios where:
    - Bin fill levels change over time (waste accumulation)
    - Routes from day N affect the state on day N+1
    - The agent must learn long-term planning strategies
    - Multi-step returns are computed when using temporal horizons

    Key Features:
    - Sequential dataset updates: Visited bins are emptied, unvisited bins accumulate waste
    - Temporal horizon support: Can look ahead H days for better decision making
    - Discounted return computation: G_t = R_t + γ * G_{t+1}
    - Compatible with temporal models (TAM) that use fill history

    Workflow:
    1. Train on day D with current bin states
    2. Generate routes using the policy
    3. Update dataset: empty visited bins, add daily waste to all bins
    4. Repeat for day D+1 with updated state

    This creates a non-stationary training environment that better reflects
    real-world waste collection dynamics.

    Attributes:
        horizon_buffer: List storing rollouts across multiple days for multi-step returns
        horizon: Number of days to accumulate before computing returns
        gamma: Discount factor for future rewards
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the TimeTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)

        self.horizon_buffer = []
        self.horizon = opts.get('temporal_horizon', 1)
        self.gamma = opts.get('gamma', 0.99) 

    def initialize_training_dataset(self):
        """
        Initialize the time-dependent training dataset.
        """
        step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(
            self.optimizer, self.day, self.problem, self.tb_logger, self.cost_weights, self.opts
        )
        self.training_dataset = training_dataset
        self.step = step
        self.data_init_args = args

    def train_day(self):
        """
        Execute training for a single day, handling horizon logic if configured.
        """
        if self.horizon > 1:
            self.train_day_horizon()
        else:
            super().train_day()
        
        # update_time_dataset is typically called during hook? 
        # But here we do it explicitly if it's day-to-day evolution.
        # Actually BaseReinforceTrainer calls 'update_context' BEFORE train_day.
        # TimeTrainer overrides update_context to call 'update_time_dataset' logic.
        pass

    def update_context(self):
        """
        Update the training context (dataset) for the next day.
        """
        if self.day > self.opts['epoch_start']:
            prev_pi = getattr(self, 'log_pi', None)
            prev_costs = getattr(self, 'log_costs', None)
            if prev_pi is not None:
                self.training_dataset = update_time_dataset(
                    self.model, self.optimizer, self.training_dataset, 
                    prev_pi, self.day - 1, self.opts, self.data_init_args, 
                    costs=prev_costs
                )

    def train_day_horizon(self):
        """
        Execute training day with temporal horizon logic (multi-step returns).
        """
        log_pi = []
        log_costs = []
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
        
        batch_results_list = []
        
        for batch_id, batch in enumerate(tqdm(day_dataloader, disable=self.opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, self.training_dataset, day_dataloader, self.opts)
            
            pi, c_dict, l_dict, batch_cost, state_tensors = self.train_batch(batch, batch_id, opt_step=False)
            batch_results_list.append(state_tensors)
            
            if pi is not None:
                log_pi.append(pi.detach().cpu())
            log_costs.append(batch_cost.detach().cpu())
            
            self.step += 1
            if pi is not None:
                 current_batch_size = pi.size(0)
            else:
                 current_batch_size = self.opts['batch_size']
            daily_total_samples += current_batch_size

        self.horizon_buffer.append(batch_results_list)

        if (self.day + 1) % self.horizon == 0:
            self.accumulate_and_update()
            
        day_duration = time.time() - start_time
        
        self.daily_loss = daily_loss 
        self.log_pi = log_pi
        self.log_costs = log_costs
        self.day_duration = day_duration
        self.daily_total_samples = daily_total_samples

    def accumulate_and_update(self):
        """
        Compute discounted returns and perform update using accumulated horizon buffer.
        """
        num_days = len(self.horizon_buffer)
        if num_days == 0: 
            return
        num_batches = len(self.horizon_buffer[0])
        
        total_loss = torch.tensor(0.0, device=self.opts['device'])
        
        for b_id in range(num_batches):
            try:
                costs_per_day = [self.horizon_buffer[d][b_id]['cost'] for d in range(num_days)]
            except IndexError:
                continue
            
            returns_per_day = []
            R = torch.zeros_like(costs_per_day[-1])
            
            for t in range(num_days - 1, -1, -1):
                R = costs_per_day[t] + self.gamma * R
                returns_per_day.insert(0, R)
            
            for t in range(num_days):
                state = self.horizon_buffer[t][b_id]
                log_prob = state['log_likelihood']
                bl_val = state['bl_val']
                entropy = state['entropy']
                im_loss = state['imitation_loss']
                im_weight = state['curr_imitation_weight']
                
                G_t = returns_per_day[t]
                
                if bl_val is not None:
                    adv = G_t - bl_val
                else:
                    adv = G_t 
                
                reinforce_loss = (adv * log_prob).mean()
                entropy_loss = -self.opts.get('entropy_weight', 0) * entropy.mean() if entropy is not None else 0.0
                
                bl_loss = 0.0
                if self.opts['baseline'] is not None and isinstance(bl_val, torch.Tensor) and bl_val.requires_grad:
                     bl_loss = 0.5 * ((bl_val - G_t) ** 2).mean()
                
                step_loss = reinforce_loss + entropy_loss + bl_loss + im_weight * im_loss
                total_loss = total_loss + step_loss
        
        loss_final = total_loss / (num_days * num_batches)
        
        self.optimizer.zero_grad()
        loss_final.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0), norm_type=2)
        self.optimizer.step()
        
        self.horizon_buffer = []


class RWATrainer(StandardTrainer):
    """
    Reward Weight Adaptation (RWA) Trainer.

    Uses a meta-learning model (RNN or other) to dynamically adapt cost function weights
    during training. The meta-model observes training performance and adjusts weights to
    optimize for better solutions.

    The RewardWeightOptimizer learns to predict optimal weight configurations based on:
    - Historical performance metrics
    - Current cost distribution
    - Training dynamics

    This approach automates the hyperparameter tuning of cost weights, adapting them
    throughout training rather than keeping them fixed.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the RWATrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = RewardWeightOptimizer(num_objectives=len(cost_weights))

    def setup_meta_learner(self):
        """
        Initialize the Reward Weight Optimizer (RWO) meta-model.
        """
        model_class = None
        if self.opts['rwo_model'] == 'rnn':
            model_class = WeightAdjustmentRNN
             
        len_weights = len(self.cost_weights.keys())
        min_weights = self.opts['meta_range'][0] * len_weights
        max_weights = self.opts['meta_range'][1] * len_weights
        
        self.weight_optimizer = RewardWeightOptimizer(
            model_class=model_class,
            initial_weights=self.cost_weights,
            history_length=self.opts.get('meta_history', 10),
            hidden_size=self.opts.get('mrl_embedding_dim', 64),
            lr=self.opts.get('mrl_lr', 0.001),
            device=self.opts['device'],
            meta_batch_size=self.opts.get('mrl_batch_size', 8),
            min_weights=min_weights,
            max_weights=max_weights,
            meta_optimizer=self.opts.get('rwa_optimizer', 'adam')
        )

    def update_context(self):
        """
        Update context hook (no-op for RWA).
        """
        pass

class ContextualBanditTrainer(TimeTrainer):
    """
    Contextual Bandit Trainer for dynamic weight selection.

    Uses a multi-armed bandit approach to select from a discrete set of weight configurations.
    The bandit algorithm (e.g., UCB, Thompson Sampling) learns which weight configurations
    work best in different contexts/situations.

    Unlike RWA which learns continuous weight adaptations, this trainer:
    - Maintains a fixed portfolio of weight configurations
    - Selects one configuration per day based on exploration-exploitation tradeoff
    - Updates selection probabilities based on observed rewards

    Combines with TimeTrainer to handle time-dependent training scenarios.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the ContextualBanditTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = None

    def setup_meta_learner(self):
        """
        Initialize the Contextual Bandit weight optimizer.
        """
        self.weight_optimizer = WeightContextualBandit(
            num_weight_configs=10, 
            initial_weights=self.cost_weights
            # Epsilon and decay rate handled in update call or defaults
        )

    def update_context(self):
        """
        Update weights via Bandit proposal.
        """
        if self.weight_optimizer:
             weights = self.weight_optimizer.propose_weights(context=None)
             self.cost_weights.update(weights)

    def process_feedback(self):
        """
        Provide feedback (reward) to the Bandit optimizer.
        """
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            self.weight_optimizer.feedback(reward, metrics=None, day=self.day)

class TDLTrainer(StandardTrainer):
    """
    Task-Dependent Learning (TDL) Trainer.

    Manages cost weights adaptively based on task characteristics using a CostWeightManager.
    The manager adjusts weights to balance multiple objectives dynamically during training.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the TDLTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = CostWeightManager(num_objectives=len(cost_weights))

class MORLTrainer(StandardTrainer):
    """
    Multi-Objective Reinforcement Learning (MORL) Trainer.

    Optimizes for multiple objectives simultaneously using a MORLWeightOptimizer.
    Learns to find Pareto-optimal solutions that balance competing objectives
    (e.g., minimizing distance vs. minimizing overflows vs. maximizing waste collected).
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the MORLTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = MORLWeightOptimizer(num_objectives=len(cost_weights))

class HyperNetworkTrainer(TimeTrainer):
    """
    HyperNetwork Meta-Learning Trainer.

    Uses a hypernetwork to generate model parameters or weights based on task context.
    The hypernetwork learns to produce optimal configurations for different problem instances
    or environmental conditions, enabling fast adaptation to new scenarios.

    Combines meta-learning with time-dependent training for dynamic waste collection.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the HyperNetworkTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.hyper_optimizer = HypernetworkOptimizer(
            cost_weight_keys=list(cost_weights.keys()),
            constraint_value=opts.get('constraint_value', 1.0),
            device=opts['device'],
            problem=problem
        )

class HRLTrainer(StandardTrainer):
    """
    Hierarchical Reinforcement Learning (HRL) Trainer.

    Implements a two-level hierarchy:
    - Manager (GATLSTManager): Decides WHEN to dispatch collection vehicles
    - Worker (Attention Model): Decides WHICH route to take when dispatched

    The manager observes temporal bin fill patterns and outputs gating probabilities,
    while the worker focuses on solving the routing subproblem. This decomposition
    helps tackle the joint routing and scheduling challenge more effectively.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        """
        Initialize the HRLTrainer.
        """
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.hrl_manager = GATLSTManager(device=opts['device'])
