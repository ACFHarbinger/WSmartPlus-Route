import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from .modules import Normalization, ActivationFunction


class Hypernetwork(nn.Module):
    """
    Hypernetwork that generates adaptive cost weights for time-variant RL tasks.
    Takes temporal and performance metrics as input and outputs cost weights.
    """
    def __init__(self, problem, embedding_dim=16, hidden_dim=64, normalization='layer', activation='relu', learn_affine=True, bias=True):
        super(Hypernetwork, self).__init__()
        self.problem = problem
        self.is_wc = problem.NAME == 'wcrp' or problem.NAME == 'cwcvrp' or problem.NAME == 'sdwcvrp'
        self.is_vrpp = problem.NAME == 'vrpp' or problem.NAME == 'cvrpp'
        self.is_pctsp = problem.NAME == 'pctsp'
        if self.is_vrpp or self.is_wc or self.is_pctsp:
            cost_dim = 3 * 2
        else:
            cost_dim = 1 * 2

        self.time_embedding = nn.Embedding(365, embedding_dim)  # Day of year embedding

        # Combined input: metrics + time embedding
        combined_dim = cost_dim + embedding_dim
        
        self.layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, hidden_dim, bias=bias),
            Normalization(hidden_dim, normalization, learn_affine),
            ActivationFunction(activation),
            nn.Linear(hidden_dim, cost_dim, bias=bias)
        )
        
        # Output activation to ensure positive weights
        self.activation = nn.Softplus()
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, metrics, day):
        """
        Generate cost weights based on current metrics and temporal information
        
        Args:
            metrics: Tensor of performance metrics [batch_size, input_dim]
            day: Tensor of day indices [batch_size]
            
        Returns:
            cost_weights: Tensor of generated cost weights [batch_size, output_dim]
        """
        # Get time embeddings
        day_embed = self.time_embedding(day % 365)
        
        # Concatenate metrics with time embeddings
        combined = torch.cat([metrics, day_embed], dim=1)
        
        # Generate raw weights
        raw_weights = self.layers(combined)
        
        # Apply activation and normalize to sum to constraint
        weights = self.activation(raw_weights)
        
        return weights


class HypernetworkOptimizer:
    """
    Manages the hypernetwork training and integration with the main RL training loop.
    """
    def __init__(self, cost_weight_keys, constraint_value, device, lr=1e-4, buffer_size=100):
        self.input_dim = 6  # [efficiency, overflows, kg, km, kg_lost, day_progress]
        self.output_dim = len(cost_weight_keys)
        self.cost_weight_keys = cost_weight_keys
        self.constraint_value = constraint_value
        self.device = device
        
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
    
    def update_buffer(self, metrics, day, weights, performance):
        """Add experience to buffer"""
        self.buffer.append({
            'metrics': metrics,
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
    
    def train(self, epochs=10):
        """Train hypernetwork on buffered experiences"""
        if len(self.buffer) < 10:  # Need minimum samples to train
            return
            
        self.hypernetwork.train()
        
        for _ in range(epochs):
            # Sample minibatch
            indices = torch.randperm(len(self.buffer))[:min(16, len(self.buffer))]
            
            metrics_batch = torch.stack([self.buffer[i]['metrics'] for i in indices]).to(self.device)
            day_batch = torch.tensor([self.buffer[i]['day'] for i in indices], dtype=torch.long).to(self.device)
            weights_batch = torch.stack([self.buffer[i]['weights'] for i in indices]).to(self.device)
            performance_batch = torch.tensor([self.buffer[i]['performance'] for i in indices]).to(self.device)
            
            # Generate predictions
            pred_weights = self.hypernetwork(metrics_batch, day_batch)
            
            # Normalize to constraint sum
            pred_weights_sum = pred_weights.sum(dim=1, keepdim=True)
            pred_weights = pred_weights * (self.constraint_value / pred_weights_sum)
            
            # Calculate loss (combine performance targeting and weight mimicking)
            best_perf_idx = performance_batch.argmin()
            target_weights = weights_batch[best_perf_idx].unsqueeze(0).expand_as(pred_weights)
            
            # Loss is MSE to best weights, weighted by relative performance
            perf_diff = (performance_batch - performance_batch.min()) / (performance_batch.max() - performance_batch.min() + 1e-8)
            perf_weights = 1.0 - perf_diff.unsqueeze(1).expand_as(pred_weights)
            
            loss = F.mse_loss(pred_weights, target_weights, reduction='none') * perf_weights
            loss = loss.mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def get_weights(self, all_costs, day, default_weights):
        """
        Generate optimized weights based on current metrics
        
        Args:
            all_costs: Dictionary of current cost values
            day: Current day (int)
            default_weights: Default weights to use if hypernetwork not ready
            
        Returns:
            dict: Dictionary of optimized cost weights
        """
        # If not enough experience, use default weights
        if len(self.buffer) < 5:
            return default_weights
        
        # Prepare input metrics
        with torch.no_grad():
            self.hypernetwork.eval()
            
            # Extract and normalize metrics
            overflows = torch.mean(all_costs['overflows'].float()).item()
            kg = torch.mean(all_costs['kg']).item()
            km = torch.mean(all_costs['km']).item()
            
            # Calculate derived metrics
            efficiency = kg / (km + 1e-8)
            kg_lost = all_costs.get('kg_lost', torch.tensor(0.0)).mean().item()
            day_progress = day / 365.0  # Normalized day of year
            
            # Combine metrics
            metrics = torch.tensor(
                [efficiency, overflows, kg, km, kg_lost, day_progress], 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            day_tensor = torch.tensor([day], dtype=torch.long).to(self.device)
            
            # Generate weights
            weights = self.hypernetwork(metrics, day_tensor).squeeze(0)
            
            # Normalize to constraint sum
            weights_sum = weights.sum()
            normalized_weights = weights * (self.constraint_value / weights_sum)
            
            # Convert to dictionary
            weights_dict = {
                key: normalized_weights[i].item() 
                for i, key in enumerate(self.cost_weight_keys)
            }
            
            return weights_dict


def train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Enhanced training process using a hypernetwork for adaptive cost weight optimization
    """
    step, is_cuda, training_dataset = prepare_epoch(optimizer, 0, problem, tb_logger, opts)
    
    # Initialize hypernetwork optimizer
    hyperopt = HypernetworkOptimizer(
        cost_weight_keys=list(cost_weights.keys()),
        constraint_value=opts['constraint'],
        device=opts['device'],
        lr=opts.get('hyper_lr', 1e-4),
        buffer_size=opts.get('hyper_buffer_size', 100)
    )
    
    if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
        training_dataset.fill_history = torch.zeros((opts['epoch_size'], opts['graph_size'], opts['temporal_horizon']))
        training_dataset.fill_history[:, :, -1] = torch.stack([instance['waste'] for instance in training_dataset.data])
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
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
        metrics_tensor = torch.tensor([
            torch.mean(all_costs['kg']) / torch.mean(all_costs['km']).clamp(min=1e-8),  # efficiency
            torch.mean(all_costs['overflows'].float()),                                 # overflows
            torch.mean(all_costs['kg']),                                                # kg
            torch.mean(all_costs['km']),                                                # km
            all_costs.get('kg_lost', torch.tensor(0.0)).mean(),                         # kg_lost
            day / 365.0                                                                 # day_progress
        ], device=opts['device'])
        
        weights_tensor = torch.tensor([cost_weights[key] for key in hyperopt.cost_weight_keys], device=opts['device'])
        
        hyperopt.update_buffer(
            metrics=metrics_tensor,
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
            print("Default weights:", {k: f"{v:.2f}" for k, v in default_weights.items()})
            print("Hypernetwork weights:", {k: f"{v:.2f}" for k, v in adaptive_weights.items()})
            
            # Use hypernetwork weights
            cost_weights = adaptive_weights
            
            # Log to tensorboard
            if tb_logger is not None:
                for k, v in cost_weights.items():
                    tb_logger.add_scalar(f'hyper_weights/{k}', v, step)
        
        log_pi = torch.stack(log_pi).contiguous().view(-1, log_pi[0].size(1))
        training_dataset = update_day(model, optimizer, training_dataset, log_pi, day+1, opts)
    
    return

opts['hyper_lr'] = 1e-4
opts['hyper_buffer_size'] = 100
opts['hyper_epochs'] = 10
opts['use_hypernetwork'] = True