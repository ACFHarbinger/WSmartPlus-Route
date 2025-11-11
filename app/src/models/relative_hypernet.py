import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque


class RelativeMetricsTracker:
    """
    Tracks relative improvements in metrics over time windows to provide
    more stable signals for hyperparameter optimization.
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.metrics_history = {
            'overflows': deque(maxlen=window_size),
            'kg': deque(maxlen=window_size),
            'km': deque(maxlen=window_size),
            'kg_lost': deque(maxlen=window_size),
            'efficiency': deque(maxlen=window_size)
        }
        self.initialized = False
    
    def update(self, metrics_dict):
        """Update metrics history with new values"""
        for key, value in metrics_dict.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Calculate efficiency and add to history
        if 'kg' in metrics_dict and 'km' in metrics_dict:
            if metrics_dict['km'] > 1e-6:  # Prevent division by very small numbers
                efficiency = metrics_dict['kg'] / metrics_dict['km']
                # Cap efficiency to reasonable range
                efficiency = min(efficiency, 100.0)
            else:
                efficiency = 0.0  # If km is too small, efficiency is meaningless
            self.metrics_history['efficiency'].append(efficiency)
        
        # Mark as initialized once we have enough data
        if all(len(history) >= 3 for history in self.metrics_history.values()):
            self.initialized = True
    
    def get_relative_improvements(self):
        """
        Calculate relative improvements for all metrics compared to moving average
        Returns dict of relative changes in range [-1, 1] where:
        - Positive values indicate improvement
        - Negative values indicate regression
        - 0 indicates no change
        """
        if not self.initialized:
            return {k: 0.0 for k in self.metrics_history.keys()}
        
        improvements = {}
        
        for key, history in self.metrics_history.items():
            if len(history) < 2:
                improvements[key] = 0.0
                continue
                
            current = history[-1]
            previous_avg = sum(list(history)[:-1]) / (len(history) - 1)
            
            if previous_avg == 0:
                improvements[key] = 0.0  # No previous data to compare
                continue
                
            # Different metrics have different improvement directions
            if key in ['overflows', 'km', 'kg_lost']:
                # For these metrics, lower is better
                relative_change = (previous_avg - current) / previous_avg
            else:
                # For efficiency and kg, higher is better
                relative_change = (current - previous_avg) / previous_avg
            
            # Bound to [-1, 1] range using tanh
            improvements[key] = torch.tanh(torch.tensor(relative_change * 5.0)).item()
        
        return improvements


class Hypernetwork(nn.Module):
    """
    Hypernetwork that generates adaptive cost weights based on relative improvements
    rather than absolute metric values.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64, time_embedding_dim=16):
        super(Hypernetwork, self).__init__()
        
        self.time_embedding = nn.Embedding(365, time_embedding_dim)  # Day of year embedding
        
        # Combined input: relative metrics + time embedding
        combined_dim = input_dim + time_embedding_dim
        
        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Output activation to ensure positive weights
        self.activation = nn.Softplus()
        
        # Initialize with reasonable defaults
        self.init_weights()
    
    def init_weights(self):
        # Initialize to produce balanced weights initially
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, relative_metrics, day, baseline_weights=None):
        """
        Generate cost weights based on relative metrics and temporal information
        
        Args:
            relative_metrics: Tensor of relative performance metrics [batch_size, input_dim]
            day: Tensor of day indices [batch_size]
            baseline_weights: Optional tensor of baseline weights to use as reference
            
        Returns:
            cost_weights: Tensor of generated cost weights [batch_size, output_dim]
        """
        # Get time embeddings
        day_embed = self.time_embedding(day % 365)
        
        # Concatenate metrics with time embeddings
        combined = torch.cat([relative_metrics, day_embed], dim=1)
        
        # Generate modulation factors (how much to adjust each weight)
        modulation = self.layers(combined)
        modulation = torch.tanh(modulation)  # Range [-1, 1]
        
        if baseline_weights is not None:
            # Apply modulation to baseline weights
            # +1 means double the weight, -1 means cut in half
            adjustment_factor = 1.0 + 0.5 * modulation  # Range [0.5, 1.5]
            weights = baseline_weights * adjustment_factor
        else:
            # Without baseline weights, directly generate weights
            weights = self.activation(modulation + 1.0)  # Shift to [0, 2] range
        
        return weights


class RelativeImprovementHyperOptimizer:
    """
    Manages the hypernetwork training using relative improvements instead of absolute metrics.
    """
    def __init__(self, cost_weight_keys, initial_weights, constraint_value, device, 
                 lr=1e-4, buffer_size=100, adaptation_rate=0.2):
        self.input_dim = 5  # [efficiency_rel, overflows_rel, kg_rel, km_rel, kg_lost_rel]
        self.output_dim = len(cost_weight_keys)
        self.cost_weight_keys = cost_weight_keys
        self.constraint_value = constraint_value
        self.device = device
        self.adaptation_rate = adaptation_rate
        
        # Store initial weights as reference
        self.initial_weights = {k: v for k, v in initial_weights.items()}
        self.initial_weights_tensor = torch.tensor(
            [self.initial_weights[k] for k in cost_weight_keys], 
            device=device, dtype=torch.float32
        )
        
        # Create relative metrics tracker
        self.metrics_tracker = RelativeMetricsTracker(window_size=5)
        
        # Create hypernetwork
        self.hypernetwork = Hypernetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=64
        ).to(device)
        
        # Create optimizer for hypernetwork
        self.optimizer = torch.optim.Adam(self.hypernetwork.parameters(), lr=lr)
        
        # Experience buffer to train hypernetwork
        self.buffer = []
        self.buffer_size = buffer_size
        
        # Performance tracking
        self.best_performance = float('inf')
        self.best_weights = None
        
        # Trend analysis
        self.performance_history = deque(maxlen=10)
        self.weight_history = {k: deque(maxlen=10) for k in cost_weight_keys}

    def update_metrics(self, all_costs):
        """Extract and update metrics from validation results"""
        metrics = {
            'overflows': torch.mean(all_costs['overflows'].float()).item(),
            'kg': torch.mean(all_costs['kg']).item(),
            'km': torch.mean(all_costs['km']).item(),
        }
        
        if 'kg_lost' in all_costs:
            metrics['kg_lost'] = torch.mean(all_costs['kg_lost']).item()
        elif 'waste' in all_costs:
            metrics['kg_lost'] = torch.mean(all_costs['waste']).item()
        else:
            # Estimate waste if not provided
            metrics['kg_lost'] = 0.0
            
        self.metrics_tracker.update(metrics)
    
    def update_buffer(self, relative_metrics, day, weights, performance):
        """Add experience to buffer"""
        self.buffer.append({
            'relative_metrics': relative_metrics,
            'day': day,
            'weights': weights,
            'performance': performance
        })
        
        # Keep buffer at desired size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Track best performance
        if performance < self.best_performance:
            self.best_performance = performance
            self.best_weights = weights.clone()
            
        # Update performance history
        self.performance_history.append(performance)
    
    def train(self, epochs=10):
        """Train hypernetwork on buffered experiences"""
        if len(self.buffer) < 10:  # Need minimum samples to train
            return
            
        self.hypernetwork.train()
        
        for _ in range(epochs):
            # Sample minibatch
            indices = torch.randperm(len(self.buffer))[:min(16, len(self.buffer))]
            
            rel_metrics_batch = torch.stack([self.buffer[i]['relative_metrics'] for i in indices]).to(self.device)
            day_batch = torch.tensor([self.buffer[i]['day'] for i in indices], dtype=torch.long).to(self.device)
            weights_batch = torch.stack([self.buffer[i]['weights'] for i in indices]).to(self.device)
            performance_batch = torch.tensor([self.buffer[i]['performance'] for i in indices]).to(self.device)
            
            # Generate predictions
            pred_weights = self.hypernetwork(
                rel_metrics_batch, 
                day_batch,
                self.initial_weights_tensor.unsqueeze(0).expand(len(indices), -1)
            )
            
            # Normalize to constraint sum
            pred_weights_sum = pred_weights.sum(dim=1, keepdim=True)
            pred_weights = pred_weights * (self.constraint_value / pred_weights_sum)
            
            # Calculate loss (combine performance targeting and weight mimicking)
            # Focus on weights that led to good performance
            performance_rank = torch.argsort(performance_batch)
            top_weights = weights_batch[performance_rank[:max(1, len(indices)//4)]]
            
            # Target is weighted average of top performing weights
            target_weights = top_weights.mean(dim=0).unsqueeze(0).expand_as(pred_weights)
            
            # Weight MSE loss by relative performance
            perf_diff = (performance_batch - performance_batch.min()) / (performance_batch.max() - performance_batch.min() + 1e-8)
            perf_weights = 1.0 - perf_diff.unsqueeze(1).expand_as(pred_weights)
            
            weight_loss = F.mse_loss(pred_weights, target_weights, reduction='none') * perf_weights
            weight_loss = weight_loss.mean()
            
            # Add regularization to prevent extreme weight adjustments
            reg_loss = F.mse_loss(pred_weights, self.initial_weights_tensor.unsqueeze(0).expand_as(pred_weights))
            
            # Combined loss
            loss = weight_loss + 0.01 * reg_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)
            
            self.optimizer.step()
    
    def get_weights(self, all_costs, day, default_weights):
        """
        Generate optimized weights based on relative improvements
        
        Args:
            all_costs: Dictionary of current cost values
            day: Current day (int)
            default_weights: Default weights to use if hypernetwork not ready
            
        Returns:
            dict: Dictionary of optimized cost weights
        """
        # Update metrics history and get relative improvements
        self.update_metrics(all_costs)
        
        relative_improvements = self.metrics_tracker.get_relative_improvements()
        
        # If not enough history or not initialized, use default weights
        if not self.metrics_tracker.initialized:
            return default_weights
        
        # Prepare relative metrics tensor
        with torch.no_grad():
            self.hypernetwork.eval()
            
            rel_metrics = torch.tensor([
                relative_improvements.get('efficiency', 0.0),
                relative_improvements.get('overflows', 0.0),
                relative_improvements.get('kg', 0.0),
                relative_improvements.get('km', 0.0),
                relative_improvements.get('kg_lost', 0.0)
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            day_tensor = torch.tensor([day], dtype=torch.long).to(self.device)
            
            # Get baseline weights tensor
            baseline_weights = torch.tensor(
                [default_weights[k] for k in self.cost_weight_keys], 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Generate weights with reference to baseline
            weights = self.hypernetwork(rel_metrics, day_tensor, baseline_weights).squeeze(0)
            
            # Normalize to constraint sum
            weights_sum = weights.sum()
            normalized_weights = weights * (self.constraint_value / weights_sum)
            
            # Apply safety mechanisms
            # 1. Ensure no weight decreases too much from initial
            min_ratio = 0.1  # Weights shouldn't drop below 10% of initial
            initial_min = torch.tensor(
                [self.initial_weights[k] * min_ratio for k in self.cost_weight_keys],
                device=self.device
            )
            normalized_weights = torch.max(normalized_weights, initial_min)
            
            # 2. Ensure no weight increases too much from initial
            max_ratio = 3.0  # Weights shouldn't exceed 3x initial
            initial_max = torch.tensor(
                [self.initial_weights[k] * max_ratio for k in self.cost_weight_keys],
                device=self.device
            )
            normalized_weights = torch.min(normalized_weights, initial_max)
            
            # 3. Specific check for length weight - if km is very small, reduce length weight
            km_mean = torch.mean(all_costs['km']).item()
            length_idx = self.cost_weight_keys.index('length') if 'length' in self.cost_weight_keys else -1
            
            if length_idx >= 0 and km_mean < 0.01:
                # If km is suspiciously low, reduce length weight
                length_weight = normalized_weights[length_idx]
                normalized_weights[length_idx] = length_weight * 0.8  # Reduce by 20%
            
            # 4. Re-normalize after safety adjustments
            weights_sum = normalized_weights.sum()
            normalized_weights = normalized_weights * (self.constraint_value / weights_sum)
            
            # Convert to dictionary
            weights_dict = {
                key: normalized_weights[i].item() 
                for i, key in enumerate(self.cost_weight_keys)
            }
            
            # Update history for this set of weights
            for k, v in weights_dict.items():
                self.weight_history[k].append(v)
            
            return weights_dict


def train_over_time_with_relative_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Enhanced training process using a hypernetwork with relative improvement metrics
    """
    step, is_cuda, training_dataset = prepare_epoch(optimizer, 0, problem, tb_logger, opts)
    
    # Initialize hypernetwork optimizer with initial weights as reference
    hyperopt = RelativeImprovementHyperOptimizer(
        cost_weight_keys=list(cost_weights.keys()),
        initial_weights=cost_weights.copy(),
        constraint_value=opts['constraint'],
        device=opts['device'],
        lr=opts.get('hyper_lr', 1e-4),
        buffer_size=opts.get('hyper_buffer_size', 100),
        adaptation_rate=opts.get('adaptation_rate', 0.2)
    )
    
    if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
        training_dataset.fill_history = torch.zeros((opts['epoch_size'], opts['graph_size'], opts['temporal_horizon']))
        training_dataset.fill_history[:, :, -1] = torch.stack([instance['waste'] for instance in training_dataset.data])
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
    # Store initial weights for reference
    initial_weights = cost_weights.copy()
    
    for day in range(opts['epoch_start'], opts['epoch_start'] + opts['n_epochs']):
        log_pi = []
        training_dataloader = torch.utils.data.DataLoader(
            baseline.wrap_dataset(training_dataset), batch_size=opts['batch_size'], pin_memory=True)
        
        start_time = time.time()
        
        if scaler is None:
            for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
                if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
                    batch_idx = get_batch_indices(batch_id, training_dataloader, len(training_dataset))
                    batch['fill_history'] = training_dataset.fill_history[batch_idx]
                
                if opts['encoder'] in ['gac', 'tgc'] and opts['focus_graph'] is not None:
                    batch['edges'] = training_dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
                
                pi = train_batch_reinforce(model, optimizer, baseline, day, batch_id, step, batch, tb_logger, cost_weights, opts)
                log_pi.append(pi)
                step += 1
        else:
            for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
                if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
                    batch_idx = get_batch_indices(batch_id, training_dataloader, len(training_dataset))
                    batch['fill_history'] = training_dataset.fill_history[batch_idx]
                
                if opts['encoder'] in ['gac', 'tgc'] and opts['focus_graph'] is not None:
                    batch['edges'] = training_dataset.edges.unsqueeze(0).expand(torch.cuda.device_count(), -1, -1)
                
                pi = train_batch_scaler_reinforce(model, optimizer, baseline, scaler, day, batch_id, step, batch, tb_logger, cost_weights, opts)
                log_pi.append(pi)
                step += 1
        
        epoch_duration = time.time() - start_time
        
        # Run validation and collect validation metrics
        default_weights = cost_weights
        cost_weights, avg_cost, all_costs = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, day, step, epoch_duration, tb_logger, is_cuda, cost_weights, opts)
        
        # Update hypernetwork buffer with this experience
        relative_metrics = torch.tensor([
            hyperopt.metrics_tracker.get_relative_improvements().get('efficiency', 0.0),
            hyperopt.metrics_tracker.get_relative_improvements().get('overflows', 0.0),
            hyperopt.metrics_tracker.get_relative_improvements().get('kg', 0.0),
            hyperopt.metrics_tracker.get_relative_improvements().get('km', 0.0),
            hyperopt.metrics_tracker.get_relative_improvements().get('kg_lost', 0.0)
        ], device=opts['device'])
        
        weights_tensor = torch.tensor([cost_weights[key] for key in hyperopt.cost_weight_keys], device=opts['device'])
        
        hyperopt.update_buffer(
            relative_metrics=relative_metrics,
            day=day,
            weights=weights_tensor,
            performance=avg_cost.item()
        )
        
        # Train hypernetwork on collected experiences
        hyperopt.train(epochs=opts.get('hyper_epochs', 10))
        
        # Get hypernetwork-generated weights for next iteration
        if opts.get('use_hypernetwork', True) and day > opts['epoch_start'] + 5:
            adaptive_weights = hyperopt.get_weights(all_costs, day, default_weights)
            
            # Log comparison of weights
            print("\nWeight comparison:")
            print("Initial weights:", {k: f"{v:.2f}" for k, v in initial_weights.items()})
            print("Current weights:", {k: f"{v:.2f}" for k, v in default_weights.items()})
            print("Hypernetwork weights:", {k: f"{v:.2f}" for k, v in adaptive_weights.items()})
            
            # Log relative improvements
            rel_improvements = hyperopt.metrics_tracker.get_relative_improvements()
            print("\nRelative improvements:")
            for k, v in rel_improvements.items():
                print(f"- {k}: {v:.4f} ({'+' if v > 0 else ''}{v*100:.1f}%)")
            
            # Use hypernetwork weights
            cost_weights = adaptive_weights
            
            # Log to tensorboard
            if tb_logger is not None:
                for k, v in cost_weights.items():
                    tb_logger.add_scalar(f'hyper_weights/{k}', v, step)
                
                # Also log relative improvements
                for k, v in rel_improvements.items():
                    tb_logger.add_scalar(f'relative_improvement/{k}', v, step)
        
        log_pi = torch.stack(log_pi).contiguous().view(-1, log_pi[0].size(1))
        training_dataset = update_day(model, optimizer, training_dataset, log_pi, day+1, opts)
    
    return