import time
import torch
import numpy as np

from abc import ABC
from tqdm import tqdm
from logic.src.utils.functions import move_to
from logic.src.utils.log_utils import log_epoch
from logic.src.pipeline.reinforcement_learning.core.epoch import (
    prepare_epoch, prepare_batch, prepare_time_dataset,
    complete_train_pass, update_time_dataset, set_decode_type,
)
from logic.src.utils.visualize_utils import visualize_epoch
from logic.src.pipeline.reinforcement_learning.core.post_processing import local_search_2opt_vectorized
from logic.src.pipeline.reinforcement_learning.meta import (
    RewardWeightOptimizer, WeightContextualBandit, 
    CostWeightManager, MORLWeightOptimizer
)
from logic.src.pipeline.reinforcement_learning.policies.hgs_vectorized import VectorizedHGS
from logic.src.models import WeightAdjustmentRNN, HypernetworkOptimizer, GATLSTManager


class BaseReinforceTrainer(ABC):
    """
    Abstract base trainer implementing the Template Method pattern for REINFORCE training loops.
    """
    def __init__(self, model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
        self.model = model
        self.optimizer = optimizer
        self.baseline = baseline
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.val_dataset = val_dataset
        self.problem = problem
        self.tb_logger = tb_logger
        self.cost_weights = cost_weights
        self.opts = opts
        
        self.step = 0
        self.day = opts['epoch_start']
        self.weight_optimizer = None # Meta-learner

    def setup_meta_learner(self):
        """
        Initialize the meta-learner if needed.
        """
        pass

    def update_context(self):
        """
        Update context/weights before the day/epoch starts.
        """
        pass

    def process_feedback(self):
        """
        Provide feedback to the meta-learner after training step/day.
        """
        pass

    def train(self):
        """
        Main training loop (Template Method).
        """
        self.setup_meta_learner()
        
        # Initial weight synchronization if meta-learner exists
        if self.weight_optimizer:
            if hasattr(self.weight_optimizer, 'propose_weights'):
                 pass
            elif hasattr(self.weight_optimizer, 'get_current_weights'):
                 weights = self.weight_optimizer.get_current_weights()
                 self.cost_weights.update(weights)

        # Initialize dataset 
        self.initialize_training_dataset()

        # Training loop
        max_days = self.opts['epoch_start'] + self.opts['n_epochs']
        while self.day < max_days:
            # Hook: Pre-day updates
            self.update_context()

            # Train for a single day/epoch
            self.train_day()
            
            # Hook: Post-day processing
            self.post_day_processing()
            self.process_feedback()

            self.day += 1
            if self._should_stop():
                break

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
            
            pi, c_dict, l_dict, batch_cost = self.train_batch(batch, batch_id)
            
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

    def train_batch(self, batch, batch_id):
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
                        
                        hgs = VectorizedHGS(dist_matrix, demands, vehicle_capacity, device=self.opts['device'])
                        
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
                            # Actually, we should check unique count too?
                            # For speed, just check length.
                            if len(giant) == expected_len:
                                pi_giant_list.append(giant)
                                valid_hgs_indices.append(i)
                            else:
                                # Fallback or keep empty placeholder?
                                # We can't batch mix valid/invalid easily in HGS.
                                pass
                        
                        expert_pi = torch.zeros((pi.size(0), pi.size(1)), dtype=torch.long, device=self.opts['device'])
                        if len(valid_hgs_indices) > 0:
                            # Convert list of numpy arrays to a single numpy array first for performance (avoid warning)
                            giant_tours_np = torch.from_numpy(np.array(pi_giant_list)).to(self.opts['device'])
                            
                            # Filter demands/dist if they vary per batch (not shared)
                            # If dist_matrix is (B, ...), we need to subset.
                            hgs_dist = dist_matrix
                            if dist_matrix.size(0) == pi.size(0):
                                hgs_dist = dist_matrix[valid_hgs_indices]
                            elif dist_matrix.size(0) == 1 and pi.size(0) > 1:
                                # Keep shared
                                hgs_dist = dist_matrix
                                
                            hgs_demands = demands
                            if demands.size(0) == pi.size(0):
                                hgs_demands = demands[valid_hgs_indices]
                                
                            # Re-init HGS with correct subset size/device if needed?
                            # VectorizedHGS uses the passed dist/demands.
                            hgs_solver = VectorizedHGS(hgs_dist, hgs_demands, vehicle_capacity, device=self.opts['device'])
                            
                            # Solve HGS
                            try:
                                expert_pi_valid, _ = hgs_solver.solve(giant_tours_np)
                                
                                # Scatter back
                                # expert_pi_valid is (B_sub, L_sub)
                                for idx, batch_idx in enumerate(valid_hgs_indices):
                                    row = expert_pi_valid[idx]
                                    # row might be padded
                                    # Copy into expert_pi
                                    # Ensure size fits
                                    copy_len = min(len(row), expert_pi.size(1))
                                    expert_pi[batch_idx, :copy_len] = row[:copy_len]
                            except Exception as e:
                                pass

                        # HGS returns routes [0, ..., 0].
                        # So we should strip the first 0.
                        if expert_pi.size(1) > 0 and expert_pi[0, 0] == 0:
                            expert_pi = expert_pi[:, 1:]
                
                else: 
                    # Default 2-opt
                    if dist_matrix is not None:
                         with torch.no_grad():
                             pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=pi.device), pi], dim=1)
                             pi_opt_with_depot = local_search_2opt_vectorized(pi_with_depot, dist_matrix, self.opts['two_opt_max_iter'])
                             expert_pi = pi_opt_with_depot[:, 1:]
                             if expert_pi.size(1) > 0:
                                 invalid_start = (expert_pi[:, 0] == 0)
                                 if invalid_start.any():
                                     expert_pi = torch.where(invalid_start.unsqueeze(-1), pi, expert_pi)

                if expert_pi is not None:
                    _, log_likelihood_opt, _, _, _ = self.model(x, cost_weights=self.cost_weights, return_pi=False, mask=mask, expert_pi=expert_pi, kl_loss=True)
                    imitation_loss = -log_likelihood_opt.mean()
            loss = reinforce_loss.mean() + bl_loss.mean() + entropy_loss.mean() + curr_imitation_weight * imitation_loss
            loss = loss / self.opts['accumulation_steps']
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (batch_id + 1) % self.opts['accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0), norm_type=2)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.get('max_grad_norm', 1.0), norm_type=2)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            l_dict = {
                'loss': loss.item(),
                'nll': -log_likelihood.mean().item(),
                'reinforce_loss': reinforce_loss.mean().item(),
                'baseline_loss': bl_loss.mean().item(),
                'imitation_loss': imitation_loss.item() if isinstance(imitation_loss, torch.Tensor) else 0.0
            }
            
            if self.scaler is not None:
                autocast_context.__exit__(None, None, None)
                
            return pi, c_dict, l_dict, cost.mean()
            
        except Exception as e:
            if self.scaler is not None: autocast_context.__exit__(None, None, None)
            raise e

    def post_day_processing(self):
        log_epoch(('day', self.day), list(self.daily_loss.keys()), self.daily_loss, self.opts)
        
        # Visualization Hook
        if self.opts.get('visualize_step', 0) > 0 and (self.day + 1) % self.opts['visualize_step'] == 0:
            visualize_epoch(self.model, self.problem, self.opts, self.day, tb_logger=self.tb_logger)
            
        _ = complete_train_pass(
            self.model, self.optimizer, self.baseline, self.lr_scheduler, self.val_dataset,
            self.day, self.step, self.day_duration, self.tb_logger, self.cost_weights, self.opts, 
            manager=self.weight_optimizer
        )

    def _should_stop(self):
        return False


class TimeTrainer(BaseReinforceTrainer):
    """
    Base trainer for time-based training (evolves over time/days).
    """
    def initialize_training_dataset(self):
        step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(
            self.optimizer, self.day, self.problem, self.tb_logger, self.cost_weights, self.opts
        )
        self.training_dataset = training_dataset
        self.step = step
        self.data_init_args = args

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


class RWATrainer(TimeTrainer):
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


class ContextualBanditTrainer(TimeTrainer):
    def setup_meta_learner(self):
        self.weight_optimizer = WeightContextualBandit(
            initial_weights=self.cost_weights,
            n_arms=self.opts.get('mrl_batch_size', 5), 
            epsilon=self.opts.get('epsilon', 0.1),
            decay_rate=self.opts.get('decay_rate', 0.99)
        )

    def update_context(self):
        super().update_context()
        if self.weight_optimizer:
             weights = self.weight_optimizer.propose_weights(context=None)
             self.cost_weights.update(weights)

    def process_feedback(self):
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            self.weight_optimizer.feedback(reward, metrics=None, day=self.day)


class TDLTrainer(TimeTrainer):
    def setup_meta_learner(self):
        self.weight_optimizer = CostWeightManager(
            initial_weights=self.cost_weights,
            learning_rate=self.opts.get('mrl_lr', 0.01)
        )
    
    def update_context(self):
        super().update_context()
        if self.weight_optimizer:
            weights = self.weight_optimizer.propose_weights()
            self.cost_weights.update(weights)

    def process_feedback(self):
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            metrics = {k: sum(v)/len(v) if len(v)>0 else 0 for k, v in self.daily_loss.items()}
            self.weight_optimizer.feedback(reward, metrics, day=self.day)


class MORLTrainer(TimeTrainer):
    def setup_meta_learner(self):
        self.weight_optimizer = MORLWeightOptimizer(
             initial_weights=self.cost_weights
        )

    def update_context(self):
        super().update_context()
        if self.weight_optimizer:
            weights = self.weight_optimizer.propose_weights()
            self.cost_weights.update(weights)

    def process_feedback(self):
        if self.weight_optimizer:
            avg_cost = sum([torch.stack(c).mean().item() for c in self.log_costs]) / len(self.log_costs)
            reward = -avg_cost
            metrics = {k: sum(v)/len(v) if len(v)>0 else 0 for k, v in self.daily_loss.items()}
            self.weight_optimizer.feedback(reward, metrics, day=self.day)


class HyperNetworkTrainer(TimeTrainer):
    def setup_meta_learner(self):
        self.hyper_optimizer = HypernetworkOptimizer(
             input_dim=self.opts.get('hyper_input_dim', 10), 
             hidden_dim=64,
             output_dim=len(self.cost_weights),
             device=self.opts['device']
        )

    def update_context(self):
        super().update_context()
        # Hypernetwork logic specific placeholder
        pass


class HRLTrainer(TimeTrainer):
    """
    Hierarchical RL Trainer using GATLSTManager for gating.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PID Logic State
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.pid_target = self.opts.get('hrl_pid_target', 0.05)
        
        # GATLSTManager
        self.hrl_manager = None
        self.hrl_optimizer = None
        
        # History
        self.waste_history = None

    def setup_meta_learner(self):
        # Initialize GATLSTManager
        self.hrl_manager = GATLSTManager(
            input_dim_static=2,
            input_dim_dynamic=self.opts.get('mrl_history', 10),
            hidden_dim=self.opts.get('gat_hidden', 128),
            lstm_hidden=self.opts.get('lstm_hidden', 64),
            batch_size=self.opts.get('mrl_batch_size', 1024),
            device=self.opts['device'],
            global_input_dim=self.opts['global_input_dim'],
            critical_threshold=self.opts['hrl_threshold'],
            shared_encoder=self.model.embedder if self.opts.get('shared_encoder', True) else None
        )
        self.hrl_optimizer = torch.optim.Adam(self.hrl_manager.parameters(), lr=self.opts.get('mrl_lr', 3e-4))
        self.hrl_manager.optimizer = self.hrl_optimizer
        self.weight_optimizer = self.hrl_manager # Alias for logging compatibility

        if self.opts.get('lr_model', 1e-4) == 0:
            for param in self.model.parameters():
                param.requires_grad = False

    def initialize_training_dataset(self):
        super().initialize_training_dataset()
        # Initialize waste history
        epoch_size = self.opts['epoch_size']
        n_nodes = self.opts['graph_size']
        history_len = self.opts.get('mrl_history', 10)
        self.waste_history = torch.zeros((epoch_size, n_nodes, history_len)).to(self.opts['device'])

    def train_day(self):
        # 1. Manager decision loop
        inference_batch_size = self.opts.get('mrl_batch_size', 1024)
        n_samples = len(self.training_dataset)
        
        mask_actions_list = []
        gate_actions_list = []
        
        # We need to access data. training_dataset.data is a list of dicts.
        for i in range(0, n_samples, inference_batch_size):
            batch_indices = slice(i, min(i + inference_batch_size, n_samples))
            batch_data = self.training_dataset.data[batch_indices]
            
            static_locs = torch.stack([x['loc'] for x in batch_data]).to(self.opts['device'])
            current_waste = torch.stack([x['waste'] for x in batch_data]).to(self.opts['device'])
            
            # Update history
            if isinstance(batch_indices, slice):
                # Need explicit indices for tensor update
                idx_len = len(batch_data)
                start = batch_indices.start
                # Tensor slice
                batch_waste_history = self.waste_history[start:start+idx_len]
            else:
                 # fallback
                 pass
            
            # Roll and update
            batch_waste_history = torch.roll(batch_waste_history, shifts=-1, dims=2)
            batch_waste_history[:, :, -1] = current_waste
            self.waste_history[start:start+idx_len] = batch_waste_history
            
            # features
            critical_mask = (current_waste > self.hrl_manager.critical_threshold).float()
            critical_ratio = critical_mask.mean(dim=1, keepdim=True)
            max_current_waste = current_waste.max(dim=1, keepdim=True)[0]
            
            global_features = torch.cat([critical_ratio, max_current_waste], dim=1)
            
            with torch.no_grad():
                mask_action, gate_action, value = self.hrl_manager.select_action(
                    static_locs, batch_waste_history, global_features, target_mask=critical_mask
                )
            mask_actions_list.append(mask_action)
            gate_actions_list.append(gate_action)
            
        mask_action_full = torch.cat(mask_actions_list, dim=0)
        gate_action_full = torch.cat(gate_actions_list, dim=0)
        self.mask_action_full = mask_action_full # Store for reward calc
        
        am_mask = (mask_action_full == 0)
        gate_expanded = gate_action_full.unsqueeze(1).expand_as(am_mask)
        final_mask = am_mask | (gate_expanded == 0)
        
        for i in range(n_samples):
            self.training_dataset.data[i]['hrl_mask'] = final_mask[i].cpu()
            
        # 2. Train Worker (Base Logic)
        super().train_day()
        
        # 3. Reward Calculation & Update
        if self.daily_total_samples > 0:
            inst_route_cost = torch.cat(self.daily_loss['length'], dim=0)
            if 'overflows' in self.daily_loss:
                inst_overflow = torch.cat(self.daily_loss['overflows'], dim=0)
            else:
                inst_overflow = torch.zeros(self.daily_total_samples, device=self.opts['device'])
            
            if 'waste' in self.daily_loss:
                 inst_waste_collected = torch.cat(self.daily_loss['waste'], dim=0)
            else:
                 inst_waste_collected = torch.zeros(self.daily_total_samples, device=self.opts['device'])
                 
            # PID Logic
            current_overflow = inst_overflow.float().mean().item()
            pid_error = current_overflow - self.pid_target
            self.pid_integral += pid_error
            pid_derivative = pid_error - self.pid_prev_error
            self.pid_prev_error = pid_error
            
            kp = self.opts.get('hrl_kp', 50.0)
            ki = self.opts.get('hrl_ki', 5.0)
            kd = self.opts.get('hrl_kd', 0.0)
            adjustment = (kp * pid_error) + (ki * self.pid_integral) + (kd * pid_derivative)
            
            l_init = self.opts.get('hrl_lambda_overflow_initial', 1000.0)
            l_min = self.opts.get('hrl_lambda_overflow_min', 100.0)
            l_max = self.opts.get('hrl_lambda_overflow_max', 2000.0)
            current_lambda_overflow = max(l_min, min(l_max, l_init + adjustment))
            
            lambda_waste = self.opts.get('hrl_lambda_waste', 300.0)
            lambda_cost = self.opts.get('hrl_lambda_cost', 0.1)
            lambda_pruning = self.opts.get('hrl_lambda_pruning', 5.0)
            
            inst_pruning_penalty = (self.mask_action_full.float().sum(dim=1)) * lambda_pruning
            
            hrl_reward_tensor = (
                lambda_waste * inst_waste_collected.float().flatten() - (
                    lambda_cost * inst_route_cost.float().flatten() + 
                    current_lambda_overflow * inst_overflow.float().flatten() + 
                    inst_pruning_penalty.float().flatten()
                )
            ) * self.opts.get('hrl_reward_scale', 0.0001)
            
            self.hrl_manager.rewards.append(hrl_reward_tensor)
            
            freq = self.opts.get('mrl_step', 10)
            if len(self.hrl_manager.rewards) >= freq:
                self.hrl_manager.update(
                    lr=self.opts.get('mrl_lr', 3e-4),
                    ppo_epochs=self.opts.get('hrl_epochs', 4),
                    clip_eps=self.opts.get('hrl_clip_eps', 0.2),
                    gamma=self.opts.get('hrl_gamma', 0.95),
                    lambda_mask_aux=self.opts.get('hrl_lambda_mask_aux', 50.0),
                    entropy_coef=self.opts.get('hrl_entropy_coef', 0.2)
                )

class StandardTrainer(BaseReinforceTrainer):
    """
    Standard trainer for epoch-based training.
    """
    def update_context(self):
        self.step, self.training_dataset, _ = prepare_epoch(
            self.optimizer, self.day, self.problem, self.tb_logger, self.cost_weights, self.opts
        )
