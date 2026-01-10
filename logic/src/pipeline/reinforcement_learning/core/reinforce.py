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
    def initialize_training_dataset(self):
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
            curr_imitation_weight = self.opts.get('imitation_weight', 0.0) * (self.opts.get('imitation_decay', 1.0) ** (self.day // self.opts.get('imitation_decay_step', 1)))
            
            if curr_imitation_weight > 0 and self.opts.get('two_opt_max_iter', 0) > 0 and pi is not None:
                dist_matrix = x.get('dist', None)
                expert_pi = None
                
                if self.opts.get('imitation_mode', '2opt') == 'hgs':
                    # HGS requires demands and capacity
                    # Assuming x['waste'] is normalized demand and capacity is 1.0 (typical for RL)
                    # or recover from opts
                    demands = x.get('waste', None)
                    # Flatten demands if needed? x['waste'] is usually (B, N) or (B, N, 1) or (B, 1, N)
                    if demands is not None:
                        if demands.dim() == 3: demands = demands.squeeze(1).squeeze(1)
                        if demands.dim() == 2 and demands.size(1) == 1: demands = demands.squeeze(1)
                        
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
                            # For speed, just check length.
                            if len(giant) == expected_len:
                                pi_giant_list.append(giant)
                                valid_hgs_indices.append(i)
                            else:
                                pass # Discard invalid routes

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
                                expert_pi = torch.zeros((pi.size(0), pi.size(1)), dtype=torch.long, device=self.opts['device'])
                                for idx, batch_idx in enumerate(valid_hgs_indices):
                                    row = expert_pi_valid[idx]
                                    copy_len = min(len(row), expert_pi.size(1))
                                    expert_pi[batch_idx, :copy_len] = row[:copy_len]
                            except Exception as e:
                                pass
                                    
                        if expert_pi.size(1) > 0 and expert_pi[0, 0] == 0:
                            expert_pi = expert_pi[:, 1:]

                elif self.opts.get('imitation_mode', '2opt') == '2opt':
                    if dist_matrix is not None:
                        if dist_matrix.size(0) == 1 and pi.size(0) > 1:
                            dist_matrix = dist_matrix.expand(pi.size(0), -1, -1)
                        with torch.no_grad():
                            pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=pi.device), pi], dim=1)
                            pi_opt_with_depot = local_search_2opt_vectorized(pi_with_depot, dist_matrix, self.opts['two_opt_max_iter'])
                            expert_pi = pi_opt_with_depot[:, 1:]
                                    
                if expert_pi is not None:
                    _, expert_log_likelihood, _, _, _ = self.model(
                        x, cost_weights=self.cost_weights, 
                        return_pi=False, 
                        pad=self.opts['train_time'],
                        expert_pi=expert_pi,
                        imitation_mode=True
                    )
                    imitation_loss = -expert_log_likelihood.mean()
            
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
            
        except Exception as e:
            if self.scaler is not None: autocast_context.__exit__(None, None, None)
            raise e

class TimeTrainer(StandardTrainer):
    """
    Trainer for time-dependent Reinforcement Learning.
    Handles sequential updates of 'waste' or other time-variant features in the dataset.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        
        self.horizon_buffer = [] 
        self.horizon = opts.get('temporal_horizon', 1) 
        self.gamma = opts.get('gamma', 0.99) 

    def initialize_training_dataset(self):
        step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(
            self.optimizer, self.day, self.problem, self.tb_logger, self.cost_weights, self.opts
        )
        self.training_dataset = training_dataset
        self.step = step
        self.data_init_args = args

    def train_day(self):
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
        num_days = len(self.horizon_buffer)
        if num_days == 0: return
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
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = RewardWeightOptimizer(num_objectives=len(cost_weights))

    def setup_meta_learner(self):
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
        pass

class ContextualBanditTrainer(TimeTrainer):
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = None

    def setup_meta_learner(self):
        self.weight_optimizer = WeightContextualBandit(
            num_weight_configs=10, 
            initial_weights=self.cost_weights
            # Epsilon and decay rate handled in update call or defaults
        )

    def update_context(self):
        if self.weight_optimizer:
             weights = self.weight_optimizer.propose_weights(context=None)
             self.cost_weights.update(weights)

    def process_feedback(self):
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            self.weight_optimizer.feedback(reward, metrics=None, day=self.day)

class TDLTrainer(StandardTrainer):
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = CostWeightManager(num_objectives=len(cost_weights))

class MORLTrainer(StandardTrainer):
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.weight_optimizer = MORLWeightOptimizer(num_objectives=len(cost_weights))

class HyperNetworkTrainer(TimeTrainer):
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.hyper_optimizer = HypernetworkOptimizer(
            cost_weight_keys=list(cost_weights.keys()),
            constraint_value=opts.get('constraint_value', 1.0),
            device=opts['device'],
            problem=problem
        )

class HRLTrainer(StandardTrainer):
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        super().__init__(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        self.hrl_manager = GATLSTManager(device=opts['device'])
