import os
import sys
import time
import torch
import traceback
import pandas as pd

from tqdm import tqdm
from .meta import (
    CostWeightManager, RewardWeightOptimizer,
    WeightContextualBandit, MORLWeightOptimizer, 
)
from logic.src.models import WeightAdjustmentRNN, GATLSTManager, HypernetworkOptimizer
from logic.src.utils.functions import move_to
from logic.src.utils.log_utils import log_values, log_training, log_epoch, get_loss_stats
from logic.src.policies import local_search_2opt_vectorized
from .post_processing import post_processing_optimization
from .epoch import (
    set_decode_type, clip_grad_norms, 
    complete_train_pass, update_time_dataset,
    prepare_epoch, prepare_batch, prepare_time_dataset, 
)


def _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, day_dataset, 
                    val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts, manager=None):
    log_pi = []
    log_costs = []
    daily_total_samples = 0
    daily_loss = {key: [] for key in loss_keys}  
    day_dataloader = torch.utils.data.DataLoader(baseline.wrap_dataset(day_dataset), batch_size=opts['batch_size'], pin_memory=True)
    start_time = time.time()
    for batch_id, batch in enumerate(tqdm(day_dataloader, disable=opts['no_progress_bar'])):
        current_weights = cost_weights if weight_optimizer is None else weight_optimizer.get_current_weights()
        batch = prepare_batch(batch, batch_id, day_dataset, day_dataloader, opts)
        pi, c_dict, l_dict, batch_cost = train_batch_reinforce(model, optimizer, baseline, scaler, day, batch_id, step, batch, tb_logger, current_weights, opts, weight_optimizer)
        log_pi.append(pi.detach().cpu())
        log_costs.append(batch_cost.detach().cpu())

        step += 1
        daily_total_samples += pi.size(0)
        for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
            if key in daily_loss and isinstance(val, torch.Tensor): 
                # Ensure it's at least 1D for torch.cat later
                daily_loss[key].append(val.detach().cpu().view(-1)) 
    
    day_duration = time.time() - start_time
    table_df.loc[day] = get_loss_stats(daily_loss)
    log_epoch(('day', day), loss_keys, daily_loss, opts)      
    _ = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, 
                            day, step, day_duration, tb_logger, cost_weights, opts, manager)
    return step, log_pi, daily_loss, daily_total_samples, current_weights, log_costs


def train_reinforce_over_time_rwa(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)
    
    # Define bounds
    len_weights = len(cost_weights.keys())
    min_weights = opts['meta_range'][0] * len_weights
    max_weights = opts['meta_range'][1] * len_weights
    
    # Create meta-learner
    model_class = {
        'rnn': WeightAdjustmentRNN,
    }.get(opts['rwo_model'], None)
    assert model_class is not None, "Unknown meta learning model: {}".format(model_class)

    weight_optimizer = RewardWeightOptimizer(
        model_class=model_class,
        initial_weights=cost_weights,
        history_length=opts.get('meta_history', 10),
        hidden_size=opts.get('mrl_embedding_dim', 64),
        lr=opts.get('mrl_lr', 0.001),
        device=opts['device'],
        meta_batch_size=opts.get('mrl_batch_size', 8),
        min_weights=min_weights,
        max_weights=max_weights,
        meta_optimizer=opts.get('rwa_optimizer', 'adam')
    )
    weight_dict = weight_optimizer.get_current_weights()
    for key, value in weight_dict.items():
        cost_weights[key] = value

    # Track weight history for analysis
    weight_history_df = pd.DataFrame(columns=weight_optimizer.weight_names)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    while True:
        step, log_pi, _, _, current_weights, log_costs = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        weight_history_df.loc[day] = current_weights
        if tb_logger is not None:
            for key, value in current_weights.items():
                tb_logger.add_scalar(key, value, day)

        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break    
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args, costs=log_costs)
    log_training(loss_keys, table_df, opts)
    if weight_optimizer is not None and 'output_dir' in opts:
        weight_history_df.to_csv(os.path.join(opts['output_dir'], "weight_history_final.csv"))
    return model, None


def train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    epsilon_parameters = (opts['cb_min_epsilon'], opts['cb_epsilon_decay'])
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

    # Initialize the Contextual Bandit for weight selection
    bandit = WeightContextualBandit(
        num_days=opts['n_epochs'],
        initial_weights=cost_weights,
        distance_matrix=training_dataset.dist_matrix,
        context_features=opts.get('cb_context_features', None),
        features_aggregation=opts.get('cb_features_aggregation', 'avg'),
        exploration_strategy=opts.get('cb_exploration_method', 'greedy'),
        exploration_factor=opts.get('mrl_exploration_factor', 0.2),
        num_weight_configs=opts.get('cb_num_configs', 10),
        weight_ranges=opts.get('mrl_range', None),
        window_size=opts.get('mrl_history', 20)
    )
    bandit.set_max_feature_values()

    # For tracking daily performance metrics
    daily_rewards = []
    daily_cost_components = {}
    daily_selected_weights = []
    cost_keys = loss_keys[:loss_keys.index('total')]

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    while True:
        step, log_pi, daily_loss, daily_total_samples, current_weights, log_costs = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, bandit, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        daily_selected_weights.append(current_weights)
        if daily_total_samples > 0:
            for ckey in cost_keys:
                daily_cost_components[ckey] = daily_loss[ckey].sum().item()

            # Calculate average daily reward and cost components
            avg_daily_reward = (
                - sum(current_weights[ckey] * (daily_cost_components[ckey] / daily_total_samples) for ckey in cost_keys)
            )
            normalized_components = {ckey: daily_cost_components[ckey] / daily_total_samples for ckey in cost_keys}
            
            # Update bandit with observed reward and context and weights
            context = {f'avg_{key}': value for key, value in normalized_components.items()}
            context.update({'day': day})
            bandit_stats = bandit.update(avg_daily_reward, normalized_components, context, epsilon_parameters)
            daily_rewards.append(avg_daily_reward)
            
            # Update the global cost_weights with our selected ones
            print(f"Contextual Bandits Avg Reward: {avg_daily_reward:.4f}")
            print(f"Bandit trials: {bandit_stats['trials']}")
            print("Components: ", end='')
            for key, value in current_weights.items():
                print(f"{key}={normalized_components[key]:.4f}", end=', ')
                cost_weights[key] = value
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, costs=log_costs)
    # At the end of training, get the best configuration
    best_config = bandit.get_best_config()
    print("Best weight configuration: ", end='')
    for key, value in best_config.items():
        print(f"{key}={value:.4f}", end=', ')

    # Save performance history
    if not opts['no_tensorboard']:
        for d, reward in enumerate(daily_rewards):
            tb_logger.log_value('daily_reward', reward, d + opts['epoch_start'])
            for wkey, wval in cost_weights.items():
                tb_logger.log_value(f'weight_{wkey}', wval, d + opts['epoch_start'])
    log_training(loss_keys, table_df, opts)
    return model, None


def train_reinforce_over_time_tdl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)
    
    # Initialize cost weight manager with initial weights 
    weight_manager = CostWeightManager(
        initial_weights=cost_weights,
        learning_rate=opts.get('mrl_lr', 0.01),
        decay_rate=opts.get('tdl_lr_decay', 0.999),
        weight_ranges=opts.get('mrl_range', [0.01, 5.0]),
        window_size=opts.get('mrl_history', 10)
    )
    
    # For tracking daily performance metrics
    daily_rewards = []
    daily_cost_components = {}
    cost_keys = loss_keys[:loss_keys.index('total')]

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    while True:
        step, log_pi, daily_loss, daily_total_samples, _, log_costs = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_manager, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        if daily_total_samples > 0:
            for ckey in cost_keys:
                daily_cost_components[ckey] = daily_loss[ckey].sum().item()

            # Calculate average daily reward and cost components
            avg_daily_reward = (
                - sum([weight_manager.weights[ckey] * (daily_cost_components[ckey] / daily_total_samples)] for ckey in cost_keys)
            )
            normalized_components = {ckey: daily_cost_components[ckey] / daily_total_samples for ckey in cost_keys}
    
            # Update weights based on daily performance
            updated_weights = weight_manager.update_weights(avg_daily_reward, normalized_components)
            daily_rewards.append(avg_daily_reward)
            
            # Update the global cost_weights with our dynamically adjusted ones
            print(f"Temporal Difference Learning Avg Reward: {avg_daily_reward:.4f}")
            print("Updated weights: ", end='')
            for key, value in updated_weights.items():
                print(f"{key}={value:.4f}", end=', ')
                cost_weights[key] = value
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args, costs=log_costs)
    if not opts['no_tensorboard']:
        for d, reward in enumerate(daily_rewards):
            tb_logger.log_value('daily_reward', reward, d + opts['epoch_start'])
            for wkey, wval in cost_weights.items():
                tb_logger.log_value(f'weight_{wkey}', wval, d + opts['epoch_start'])
    log_training(loss_keys, table_df, opts)
    return model, None


def train_reinforce_over_time_morl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)
    
    # Define weight constraints
    min_weights, max_weights = {}, {}
    for wkey in cost_weights.keys():
        min_weights[wkey] = opts['mrl_range'][0]
        max_weights[wkey] = opts['mrl_range'][1]
    
    weight_optimizer = MORLWeightOptimizer(
        initial_weights=cost_weights,
        weight_names=cost_weights.keys(),
        objective_names=opts.get('morl_objectives', ['waste_efficiency', 'overflow_rate']),
        min_weights=min_weights,
        max_weights=max_weights,
        history_window=opts.get('mrl_history', 20),
        exploration_factor=opts.get('mrl_exploration_factor', 0.2),
        adaptation_rate=opts.get('morl_adaptation_rate', 0.1),
        device=opts['device']
    )
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    pareto_plot_interval = opts.get('pareto_plot_interval', 10)  # Days between Pareto front plots
    while True:
        step, log_pi, _, _, _, log_costs = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        
        # Plot Pareto front periodically
        if weight_optimizer is not None and day % pareto_plot_interval == 0:
            plot_path = os.path.join(opts.get('output_dir', './checkpoints'), f"pareto_front_day_{day}.png")
            weight_optimizer.pareto_front.plot_front(save_path=plot_path)
            if 'output_dir' in opts:
                history_df = weight_optimizer.get_weight_history_dataframe()
                history_df.to_csv(os.path.join(opts['output_dir'], f"weight_history_day_{day}.csv"))
        
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args, costs=log_costs)
    log_training(loss_keys, table_df, opts)
    if weight_optimizer is not None:
        plot_path = f"{opts.get('output_dir', './checkpoints')}/pareto_front_final.png"
        weight_optimizer.pareto_front.plot_front(save_path=plot_path)
        if 'output_dir' in opts:
            history_df = weight_optimizer.get_weight_history_dataframe()
            history_df.to_csv(os.path.join({opts['output_dir']}, "weight_history_final.csv"))
    return model, None


def train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Enhanced training process using a hypernetwork for adaptive cost weight optimization
    """
    # Initialize using time dataset preparation to get correct args for updates
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)
    
    # Initialize hypernetwork optimizer
    hyperopt = HypernetworkOptimizer(
        cost_weight_keys=list(cost_weights.keys()),
        constraint_value=opts['constraint'],
        device=opts['device'],
        lr=opts.get('hyper_lr', 1e-4),
        buffer_size=opts.get('hyper_buffer_size', 100)
    )
    
    if opts['temporal_horizon'] > 0 and opts['model'] in ['tam']:
        # This might be redundant if prepare_time_dataset handles it, but keeping for safety if specific to hypernetwork flow
        pass 
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
    # Use 'day' from prepare_time_dataset which respects epoch_start
    while day < opts['epoch_start'] + opts['n_epochs']:
        log_pi = []
        log_costs = []
        training_dataloader = torch.utils.data.DataLoader(
            baseline.wrap_dataset(training_dataset), batch_size=opts['batch_size'], pin_memory=True)
        
        start_time = time.time()
        
        for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
            batch = prepare_batch(batch, batch_id, training_dataset, training_dataloader, opts)
            
            pi, _, _, batch_cost = train_batch_reinforce(model, optimizer, baseline, scaler, day, batch_id, step, batch, tb_logger, cost_weights, opts)
            log_pi.append(pi)
            log_costs.append(batch_cost.detach().cpu())
            step += 1
        
        epoch_duration = time.time() - start_time
        
        # Run validation and collect validation metrics
        default_weights = cost_weights.copy()
        
        # Determine is_cuda for validation
        is_cuda = torch.cuda.is_available() and not opts['no_cuda']
        
        cost_weights, avg_cost, all_costs = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, day, step, epoch_duration, tb_logger, is_cuda, cost_weights, opts)
        
        # Update hypernetwork buffer with this experience
        metrics_tensor = torch.tensor([
            torch.mean(all_costs['kg']) / torch.mean(all_costs['km']).clamp(min=1e-8),  # efficiency
            torch.mean(all_costs['overflows'].float()),                                 # overflows
            torch.mean(all_costs['kg']),                                                # kg
            torch.mean(all_costs['km']),                                                # km
            all_costs.get('kg_lost', torch.tensor(0.0)).mean(),                         # kg_lost
            day / opts['n_epochs']                                                                 # day_progress
        ], device=opts['device'])
        
        weights_tensor = torch.tensor([default_weights[key] for key in hyperopt.cost_weight_keys], device=opts['device'])
        
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
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day+1, opts, args, costs=log_costs)
        day += 1
    
    return model, hyperopt


def train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    # --- Gating Mechanism HRL Logic ---
    pid_integral = 0.0
    pid_prev_error = 0.0
    pid_target = opts.get('hrl_pid_target', 0.05) 

    daily_rewards = []
    if opts.get('lr_model', 1e-4) > 0:
        model.train()
    else:
        model.eval()
    set_decode_type(model, "sampling")

    # Initialize Dataset for Day Start
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

    if opts.get('lr_model', 1e-4) == 0:
        for param in model.parameters():
            param.requires_grad = False
    
    # Initialize GATLSTManager
    hrl_manager = GATLSTManager(
        input_dim_static=2,
        input_dim_dynamic=opts.get('mrl_history', 10),
        hidden_dim=opts.get('gat_hidden', 128),
        lstm_hidden=opts.get('lstm_hidden', 64),
        batch_size=opts.get('mrl_batch_size', 1024),
        device=opts['device'],
        global_input_dim=opts['global_input_dim'],
        critical_threshold=opts['hrl_threshold'],
        shared_encoder=model.embedder if opts.get('shared_encoder', True) else None
    )
    
    # Setup Waste History Buffer (Batch, N, History)
    # We need to maintain this manually as dataset evolves
    # Initialize with zeros or initial waste
    n_nodes = opts['graph_size']
    epoch_size = opts['epoch_size']
    history_len = opts.get('mrl_history', 10)
    
    waste_history = torch.zeros((epoch_size, n_nodes, history_len)).to(opts['device'])
    
    # Optimizer for Manager
    hrl_optimizer = torch.optim.Adam(hrl_manager.parameters(), lr=opts.get('mrl_lr', 3e-4))
    hrl_manager.optimizer = hrl_optimizer
    while True:
        # Batch Manager Decision and Mask Creation
        inference_batch_size = opts.get('mrl_batch_size', 1024) # Adjustable or from opts
        n_samples = len(training_dataset)
        
        mask_actions_list = []
        gate_actions_list = []
        for i in range(0, n_samples, inference_batch_size):
            # Slice indices
            batch_indices = slice(i, min(i + inference_batch_size, n_samples))
            
            # Extract batch data
            # Using a list comprehension for slice is inefficient for huge lists but dataset.data is list
            batch_data = training_dataset.data[batch_indices] # Assuming data is list or sliceable
            
            static_locs = torch.stack([x['loc'] for x in batch_data]).to(opts['device'])
            current_waste = torch.stack([x['waste'] for x in batch_data]).to(opts['device'])
            
            # Update Waste History Slice
            # Slice waste_history tensor
            batch_waste_history = waste_history[batch_indices]
            batch_waste_history = torch.roll(batch_waste_history, shifts=-1, dims=2)
            batch_waste_history[:, :, -1] = current_waste
            
            # Write back to main history
            waste_history[batch_indices] = batch_waste_history
            
            # 1. Critical Ratio Now (% > 0.9)
            critical_mask = (current_waste > hrl_manager.critical_threshold).float()
            critical_ratio = critical_mask.mean(dim=1, keepdim=True) # (B, 1)
            
            # 2. Max Current Waste
            max_current_waste = current_waste.max(dim=1, keepdim=True)[0] # (B, 1)
            
            # Combine: (B, 2)
            global_features = torch.cat([
                critical_ratio, 
                max_current_waste, 
            ], dim=1)

            # Manager Decision
            with torch.no_grad():
                mask_action, gate_action, value = hrl_manager.select_action(
                    static_locs, batch_waste_history, global_features, 
                    target_mask=critical_mask
                )
            
            mask_actions_list.append(mask_action)
            gate_actions_list.append(gate_action)
            
        # Concatenate results
        mask_action_full = torch.cat(mask_actions_list, dim=0)
        gate_action_full = torch.cat(gate_actions_list, dim=0)
        
        # 3. Apply Decision        
        # mask_action_full: 1 = Keep, 0 = Mask
        # am_mask: True = Skip/Masked, False = Keep/Unmasked
        am_mask = (mask_action_full == 0)
        
        gate_expanded = gate_action_full.unsqueeze(1).expand_as(am_mask) # (B, N)
        final_mask = am_mask | (gate_expanded == 0)
        
        # Store in dataset for _train_single_day loop
        for i in range(n_samples):
            training_dataset.data[i]['hrl_mask'] = final_mask[i].cpu()
        
        # 4. Train Worker 
        step, log_pi, daily_loss, daily_total_samples, _, log_costs = _train_single_day(
            model, optimizer, baseline, lr_scheduler, scaler, None, training_dataset, 
            val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts, hrl_manager
        )
        
        with torch.no_grad():
            avg_gate_prob = hrl_manager.forward(static_locs, batch_waste_history, global_features)[1].softmax(-1)[:, 1].mean().item()

        # 5. Reward Calculation
        if daily_total_samples > 0:             
            # Concatenate all batches for the day
            inst_route_cost = torch.cat(daily_loss['length'], dim=0) # (TotalSamples)
            
            # Overflow: 'overflows' per instance
            if 'overflows' in daily_loss:
                inst_overflow = torch.cat(daily_loss['overflows'], dim=0) # (TotalSamples)
            else:
                inst_overflow = torch.zeros(daily_total_samples, device=opts['device'])
            
            # Collected Waste: 'waste' per instance
            if 'waste' in daily_loss:
                inst_waste_collected = torch.cat(daily_loss['waste'], dim=0) # (TotalSamples)
            else:
                inst_waste_collected = torch.zeros(daily_total_samples, device=opts['device'])
            
            # PID Lagrangian Control for OVERFLOWS (REPORT IMPLEMENTATION)
            # Dynamic Lambda based on Overflow Rate.
            # Target: 0.05 (5%).
            # If Overflows > 0.05 (Error > 0): Meaning we are failing constraints.
            # We need to PENALIZE OVERFLOWS MORE. 
            current_overflow = inst_overflow.float().mean().item()
            pid_error = current_overflow - pid_target
            
            # Update Integral
            pid_integral += pid_error
            # Update Derivative
            pid_derivative = pid_error - pid_prev_error
            pid_prev_error = pid_error
            
            # Calculate Output (Adjustment)
            kp_overflow = opts.get('hrl_kp', 50.0)
            ki_overflow = opts.get('hrl_ki', 5.0)
            kd_overflow = opts.get('hrl_kd', 0.0)
            
            adjustment = (kp_overflow * pid_error) + (ki_overflow * pid_integral) + (kd_overflow * pid_derivative)
            
            # Base restart value hrl_lambda_overflow_initial, adjust from there
            # Cap limit to hrl_lambda_overflow_max to prevent reward signal saturation
            l_init = opts.get('hrl_lambda_overflow_initial', 1000.0)
            l_min = opts.get('hrl_lambda_overflow_min', 100.0)
            l_max = opts.get('hrl_lambda_overflow_max', 2000.0)
            current_lambda_overflow = max(l_min, min(l_max, l_init + adjustment))
            
            # Mileage Recovery (Travel-Agnostic Routing)
            inst_gate_penalty = 0.0 
            
            lambda_waste = opts.get('hrl_lambda_waste', 300.0)
            lambda_cost = opts.get('hrl_lambda_cost', 0.1)
            lambda_overflow = current_lambda_overflow # DYNAMIC PID VALUE
            lambda_pruning = opts.get('hrl_lambda_pruning', 5.0)
            
            inst_pruning_penalty = (mask_action_full.float().sum(dim=1)) * lambda_pruning
            
            # hrl_reward_tensor: (TotalSamples)
            hrl_reward_tensor = (
                lambda_waste * inst_waste_collected.float().flatten() - (
                    lambda_cost * inst_route_cost.float().flatten() + 
                    lambda_overflow * inst_overflow.float().flatten() + 
                    inst_gate_penalty + 
                    inst_pruning_penalty.float().flatten()
                )
            ) * opts.get('hrl_reward_scale', 0.0001)
            
            daily_rewards.append(hrl_reward_tensor.mean().item())
            hrl_manager.rewards.append(hrl_reward_tensor) # Append the tensor
            
            # Update Manager
            freq = opts.get('mrl_step', 10)
            if len(hrl_manager.rewards) >= freq:
                loss = hrl_manager.update(
                    lr=opts.get('mrl_lr', 3e-4),
                    ppo_epochs=opts.get('hrl_epochs', 4),
                    clip_eps=opts.get('hrl_clip_eps', 0.2),
                    gamma=opts.get('hrl_gamma', 0.95),
                    lambda_mask_aux=opts.get('hrl_lambda_mask_aux', 50.0),
                    entropy_coef=opts.get('hrl_entropy_coef', 0.2)
                )
                if loss is not None and tb_logger is not None:
                    tb_logger.log_value('hrl_manager_loss', loss, step)
                    tb_logger.log_value('lambda_overflow', lambda_overflow, step)

        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args, costs=log_costs)
        
    log_training(loss_keys, table_df, opts)
    return model, hrl_manager


def train_reinforce_over_time(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    while True:
        step, log_pi, _, _, _, log_costs = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, None, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args, costs=log_costs)
    
    log_training(loss_keys, table_df, opts)
    if opts['post_processing_epochs'] > 0:
        print("Starting post-processing optimization...")
        post_processor = post_processing_optimization(
            main_model=model,
            dataset=training_dataset,
            epochs=opts['post_processing_epochs'],
            lr=opts.get('lr_post_processing', 0.001),
            efficiency_weight=opts.get('efficiency_weight', 0.8),
            overflow_weight=opts.get('overflow_weight', 0.2)
        )
        
        # Save the post-processor
        if opts.get('checkpoint_encoder', None) is not None:
            torch.save(post_processor.state_dict(), f"{opts['checkpoint_encoder']}_post_processor.pt")
    return model, None


def train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, scaler, epoch, val_dataset, problem, tb_logger, cost_weights, opts):
    step, training_dataset, loss_keys = prepare_epoch(optimizer, epoch, problem, tb_logger, cost_weights, opts)
    epoch_loss = {key: [] for key in loss_keys}
    training_dataloader = torch.utils.data.DataLoader(
        baseline.wrap_dataset(training_dataset), batch_size=opts['batch_size'], pin_memory=True)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    start_time = time.time()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
        batch = prepare_batch(batch, batch_id, training_dataset, training_dataloader, opts)
        _, c_dict, l_dict, _ = train_batch_reinforce(model, optimizer, baseline, scaler, epoch, batch_id, step, batch, tb_logger, cost_weights, opts)
        step += 1
        for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
            if key in epoch_loss and isinstance(val, torch.Tensor): epoch_loss[key].append(val.detach().cpu().view(-1)) 

    epoch_duration = time.time() - start_time
    log_epoch(('epoch', epoch), loss_keys, epoch_loss, opts)
    _ = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, 
                            epoch, step, epoch_duration, tb_logger, cost_weights, opts)    
    return model, None


def train_batch_reinforce(model, optimizer, baseline, scaler, epoch, batch_id, step, 
                          batch, tb_logger, cost_weights, opts, weight_optimizer=None):
    # Update cost_weights with optimizer recommendations if available
    if weight_optimizer is not None:
        current_weights = weight_optimizer.get_current_weights()
        for key in current_weights:
            if key in cost_weights:
                cost_weights[key] = current_weights[key]

    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts['device'])
    bl_val = move_to(bl_val, opts['device']) if bl_val is not None else None
    if scaler is not None:
        # Create autocast context manager and enter context
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast_context.__enter__()
    try:
        # Evaluate model, get costs and log probabilities
        # Pass hrl_mask if available in batch
        mask = batch.get('hrl_mask', None)
        if mask is not None:
            mask = move_to(mask, opts['device'])
            
        # Standard run on full batch
        cost, log_likelihood, c_dict, pi, entropy = model(x, cost_weights=cost_weights, return_pi=opts['train_time'], pad=opts['train_time'], mask=mask)

        # Evaluate baseline, get baseline loss if any (only for critic)
        # If POMO is used, calculate shared baseline (mean cost of all trajectories for the same instance)
        if opts.get('pomo_size', 0) > 1:
            # Reshape cost to (Batch, POMO) and calculate mean per instance
            cost_pomo = cost.view(-1, opts['pomo_size'])
            bl_val = cost_pomo.mean(dim=1, keepdim=True).expand_as(cost_pomo).reshape(-1)
            bl_loss = torch.tensor([0.0], device=opts['device']) # Shared baseline doesn't need separate critic loss
        else:
            bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
            if not isinstance(bl_loss, torch.Tensor):
                bl_loss = torch.tensor([bl_loss], device=opts['device'], dtype=torch.float)
            elif bl_loss.dim() == 0:
                bl_loss = bl_loss.unsqueeze(0)

        # Calculate standard reinforce loss
        reinforce_loss = (cost - bl_val) * log_likelihood
        
        # Entropy Regularization: maximize entropy by subtracting it from the loss
        entropy_loss = -opts.get('entropy_weight', 0.0) * entropy if entropy is not None else 0.0
        
        # Look-Ahead update (Refine and Imitate)
        imitation_loss = torch.tensor(0.0, device=opts['device'])
        curr_imitation_weight = opts.get('imitation_weight', 0.0) * (opts.get('imitation_decay', 1.0) ** (epoch // opts.get('imitation_decay_step', 1)))
        if curr_imitation_weight > 0 and opts.get('two_opt_max_iter', 0) > 0 and pi is not None:
            dist_matrix = x.get('dist', None)
            if dist_matrix is not None:
                with torch.no_grad():
                    # Prepend depot (0) to each trajectory in the batch
                    pi_with_depot = torch.cat([torch.zeros((pi.size(0), 1), dtype=torch.long, device=pi.device), pi], dim=1)
                    pi_opt_with_depot = local_search_2opt_vectorized(pi_with_depot, dist_matrix, opts['two_opt_max_iter'])
                    # Remove the starting depot
                    pi_opt = pi_opt_with_depot[:, 1:]
                    
                    # SAFEGUARD: If 2-opt produced a route starting with 0 (0->0), 
                    # it violates the conditional masking (Forced Exploration).
                    # Revert these trajectories to the original pi.
                    if pi_opt.size(1) > 0:
                        invalid_start = (pi_opt[:, 0] == 0)
                        if invalid_start.any():
                            # print(f"Reverting {invalid_start.sum()} invalid 2-opt trajectories (0->0 start)")
                            pi_opt = torch.where(invalid_start.unsqueeze(-1), pi, pi_opt)
                
                # Imitation pass (Calculates log-likelihood of refined trajectories)
                # Now using KL-Regularized Behavior Cloning
                _, log_likelihood_opt, _, _, _ = model(x, cost_weights=cost_weights, return_pi=False, mask=mask, expert_pi=pi_opt, kl_loss=True)
                
                # Model returns -KL_div for each sample in batch
                # imitation_loss logic: we want to minimize KL. 
                # log_likelihood_opt is -KL.
                # So imitation_loss = -(-KL) = KL.
                imitation_loss = -log_likelihood_opt.mean()
        
        # Total loss
        loss = reinforce_loss.mean() + bl_loss.mean() + entropy_loss.mean() + curr_imitation_weight * imitation_loss
        loss = loss / opts['accumulation_steps']
    except Exception as e:
        if scaler is not None: autocast_context.__exit__(None, None, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        traceback.print_exc(file=sys.stderr)
        print(f"Error in train_batch: {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    if loss.requires_grad:
        if scaler is None:
            # Perform backward pass
            loss.backward()
        else:
            # Exit autocast context
            autocast_context.__exit__(None, None, None)
            # Perform backward pass and unscale gradients
            scaler.scale(loss).backward()
    elif scaler is not None:
        autocast_context.__exit__(None, None, None)

    # Perform optimization step with accumulated gradients
    if step % opts['accumulation_steps'] == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts['max_grad_norm'])
        
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()
    else:
        # For non-step batches, we return 0 for both raw and clipped norms per group
        dummy_norms = [torch.tensor([0.0])] * len(optimizer.param_groups)
        grad_norms = (dummy_norms, dummy_norms)

    # Logging
    l_dict = {'nll': -log_likelihood, 'reinforce_loss': reinforce_loss, 'baseline_loss': bl_loss, 'imitation_loss': imitation_loss}
    if step % opts['log_step'] == 0: log_values(cost, grad_norms, epoch, batch_id, step, l_dict, tb_logger, opts)
    
    # Update optimizer with performance metrics if available
    if weight_optimizer is not None and c_dict is not None:
        if isinstance(weight_optimizer, MORLWeightOptimizer):
            metrics = {key: val.mean().item() if isinstance(val, torch.Tensor) else val for key, val in c_dict.items()}
            
            # Add total bins count for overflow rate calculation
            metrics['total_bins'] = opts.get('graph_size', 20)
            
            # Update weights based on performance
            weight_optimizer.update_weights(
                metrics=metrics,
                reward=-cost.mean().item(),
                day=epoch,
                step=step
            )
            
            # Log weight values
            if tb_logger is not None and step % opts['log_step'] == 0:
                for name, value in weight_optimizer.current_weights.items():
                    tb_logger.add_scalar(f'weight_{name}', value, step)
        else:
            assert isinstance(weight_optimizer, RewardWeightOptimizer)
            # Extract performance metrics
            performance_metrics = [c_dict.get(key, 0.0) for key in list(c_dict.keys())[:-1]]
            # Convert to tensor if not already
            performance_metrics = [
                p.float().mean().item() if isinstance(p, torch.Tensor) else p
                for p in performance_metrics
            ]
            # Add to history
            weight_optimizer.update_histories(performance_metrics, -cost.mean().item())
            
            # Periodically perform meta-learning update
            if step % opts.get('mrl_step', 100) == 0:
                meta_loss = weight_optimizer.meta_learning_step()
                if meta_loss is not None and tb_logger is not None:
                    tb_logger.add_scalar('meta_loss', meta_loss, step)
                    
            
            # Update weights periodically
            if step % opts.get('rwa_update_step', 500) == 0:
                updated = weight_optimizer.update_weights()
    
            if step % opts.get('mrl_step', 100) == 0 or step % opts.get('rwa_update_step', 500) == 0:
                if tb_logger is not None and (meta_loss is not None or updated is not None):
                    for i, name in enumerate(weight_optimizer.weight_names):
                        tb_logger.add_scalar(f'weight_{name}', weight_optimizer.current_weights[i].item(), step)
    return pi, c_dict, l_dict, cost.detach()
