import os
import sys
import time
import torch
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
from .meta import (
    WeightContextualBandit, MORLWeightOptimizer, 
    HRLManager, CostWeightManager, RewardWeightOptimizer,
)
from logic.src.models import WeightAdjustmentRNN, GATLSTManager
from logic.src.utils.functions import move_to
from logic.src.utils.log_utils import log_values, log_training, log_epoch, get_loss_stats
from .post_processing import post_processing_optimization
from .epoch import (
    set_decode_type, clip_grad_norms, 
    complete_train_pass, update_time_dataset,
    prepare_epoch, prepare_batch, prepare_time_dataset, 
)


def _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, day_dataset, 
                    val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts, manager=None):
    log_pi = []
    daily_total_samples = 0
    daily_loss = {key: [] for key in loss_keys}  
    day_dataloader = torch.utils.data.DataLoader(baseline.wrap_dataset(day_dataset), batch_size=opts['batch_size'], pin_memory=True)
    start_time = time.time()
    for batch_id, batch in enumerate(tqdm(day_dataloader, disable=opts['no_progress_bar'])):
        current_weights = cost_weights if weight_optimizer is None else weight_optimizer.get_current_weights()
        batch = prepare_batch(batch, batch_id, day_dataset, day_dataloader, opts)
        pi, c_dict, l_dict = train_batch_reinforce(model, optimizer, baseline, scaler, day, batch_id, step, batch, tb_logger, current_weights, opts, weight_optimizer)
        log_pi.append(pi)

        step += 1
        daily_total_samples += pi.size(0)
        for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
            if key in daily_loss and isinstance(val, torch.Tensor): 
                daily_loss[key].append(val.detach()) 

    day_duration = time.time() - start_time
    table_df.loc[day] = get_loss_stats(daily_loss)
    log_epoch(('day', day), loss_keys, daily_loss, opts)      
    _ = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, 
                            day, step, day_duration, tb_logger, cost_weights, opts, manager)
    return step, log_pi, daily_loss, daily_total_samples, current_weights


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
        step, log_pi, _, _, current_weights = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        weight_history_df.loc[day] = current_weights
        if tb_logger is not None:
            for key, value in current_weights.items():
                tb_logger.add_scalar(key, value, day)

        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break    
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)
    log_training(loss_keys, table_df, opts)
    if weight_optimizer is not None and 'output_dir' in opts:
        weight_history_df.to_csv(os.path.join(opts['output_dir'], "weight_history_final.csv"))
    return model, None


def train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    epsilon_parameters = (opts['cb_min_epsilon'], opts['cb_epsilon_decay'])
    step, training_dataset, loss_keys, table_df = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

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
        step, log_pi, daily_loss, daily_total_samples, current_weights = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, bandit, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
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
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts)
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
        step, log_pi, daily_loss, daily_total_samples, _ = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_manager, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
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
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)
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
        step, log_pi, _, _, _ = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        
        # Plot Pareto front periodically
        if weight_optimizer is not None and day % pareto_plot_interval == 0:
            plot_path = os.path.join(opts.get('output_dir', './checkpoints'), f"pareto_front_day_{day}.png")
            weight_optimizer.pareto_front.plot_front(save_path=plot_path)
            if 'output_dir' in opts:
                history_df = weight_optimizer.get_weight_history_dataframe()
                history_df.to_csv(os.path.join(opts['output_dir'], f"weight_history_day_{day}.csv"))
        
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)
    log_training(loss_keys, table_df, opts)
    if weight_optimizer is not None:
        plot_path = f"{opts.get('output_dir', './checkpoints')}/pareto_front_final.png"
        weight_optimizer.pareto_front.plot_front(save_path=plot_path)
        if 'output_dir' in opts:
            history_df = weight_optimizer.get_weight_history_dataframe()
            history_df.to_csv(os.path.join({opts['output_dir']}, "weight_history_final.csv"))
    return model, None


def train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    if opts.get('hrl_method', 'weight_manager') == 'gating_mechanism':
        # --- Gating Mechanism HRL Logic ---        
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
            # RESET daily lookahead penalty
            total_lookahead_penalty_ep = torch.zeros(epoch_size, device=opts['device'])
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
                
                # Compute Global Features
                # current_waste: (B, N)
                # Heuristic: Assume 15% daily generation
                daily_gen_est = 0.15
                horizon = 3
                
                # 1. Critical Ratio Now (% > 0.9)
                critical_mask = (current_waste > 0.9).float()
                critical_ratio = critical_mask.mean(dim=1, keepdim=True) # (B, 1)
                
                # 2. Projected Critical Ratio (in 3 days)
                # bins that WILL overflow if nothing is done
                projected_waste = current_waste + (daily_gen_est * horizon)
                projected_critical_mask = (projected_waste > 1.0).float()
                critical_ratio_proj = projected_critical_mask.mean(dim=1, keepdim=True) # (B, 1)
                
                # 3. Projected Average Overflow
                projected_overflow_val = F.relu(projected_waste - 1.0)
                avg_proj_overflow = projected_overflow_val.mean(dim=1, keepdim=True) # (B, 1)
                
                # Combine: (B, 3)
                global_features = torch.cat([critical_ratio, critical_ratio_proj, avg_proj_overflow], dim=1)
                
                # Accumulate Penalty
                # avg_proj_overflow is (B, 1)
                total_lookahead_penalty_ep[batch_indices] += avg_proj_overflow.squeeze(1)

                # Expert Forcing (Strategy 4)
                # ---------------------------
                # Epsilon Decay: 1.0 -> 0.05 over training
                # Assumes 50 epochs total.
                progress = max(0, (day - opts['epoch_start']) / opts['n_epochs'])
                epsilon = max(0.05, 1.0 - (progress * 2.0)) # Decay to 0 over first 50% epochs
                
                # Expert Trigger: If > 10% bins are critical (>90%), we MUST route.
                # critical_ratio is (B, 1)
                expert_trigger_mask = (critical_ratio > 0.1).float().squeeze(1) # (B)
                
                # Apply Epsilon Greedy
                batch_size = critical_ratio.size(0)
                rand_vals = torch.rand(batch_size, device=opts['device'])
                force_mask = (rand_vals < epsilon) & (expert_trigger_mask > 0)
                
                # Construct forced action tensor
                # -1: No force
                #  1: Force Route
                force_action = torch.full((batch_size,), -1, dtype=torch.long, device=opts['device'])
                force_action[force_mask] = 1 # Force Route
                
                # Manager Decision
                with torch.no_grad():
                    mask_action, gate_action, value = hrl_manager.select_action(
                        static_locs, batch_waste_history, global_features, 
                        force_action=force_action
                    )
                
                mask_actions_list.append(mask_action)
                gate_actions_list.append(gate_action)
                
                # Step 5: Reward Calculation
                # --------------------------
                
            # Concatenate results
            mask_action_full = torch.cat(mask_actions_list, dim=0)
            gate_action_full = torch.cat(gate_actions_list, dim=0)
            
            # 3. Apply Decision
            am_mask = (mask_action_full == 0) # True where we want to skip
            gate_expanded = gate_action_full.unsqueeze(1).expand_as(am_mask) # (B, N)
            final_mask = am_mask | (gate_expanded == 0)
            
            # Store in dataset for _train_single_day loop
            for i in range(n_samples):
                training_dataset.data[i]['hrl_mask'] = final_mask[i].cpu()
            
            # 4. Train Worker 
            step, log_pi, daily_loss, daily_total_samples, _ = _train_single_day(
                model, optimizer, baseline, lr_scheduler, scaler, None, training_dataset, 
                val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts, hrl_manager
            )
            
            # DEBUG: Log Manager performance for the day
            with torch.no_grad():
                avg_gate_prob = hrl_manager.forward(static_locs, batch_waste_history, global_features)[1].softmax(-1)[:, 1].mean().item()
                force_count = force_mask.sum().item()
                print(f"DEBUG Day {day}: Avg Gate Prob (last batch)={avg_gate_prob:.4f}, Forced={force_count}/{batch_size}, Epsilon={epsilon:.3f}")

            # 5. Reward Calculation
            if daily_total_samples > 0:             
                # Route Cost: 'length' per instance
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
                
                # Future Risk: Lookahead Penalty (already computed per instance in total_lookahead_penalty_ep)
                inst_lookahead = total_lookahead_penalty_ep # (TotalSamples)
                
                # Coefficients
                lambda_waste = 10.0
                lambda_cost = 1.0
                lambda_overflow = 50.0 # Heavy penalty for real overflows
                lambda_lookahead = 20.0 # Dense penalty for projected overflows
                
                # Per-Instance Reward Tensor: (TotalSamples)
                hrl_reward_tensor = (lambda_waste * inst_waste_collected.float() - (lambda_cost * inst_route_cost.float() + lambda_overflow * inst_overflow.float() + lambda_lookahead * inst_lookahead.float())) * 0.01
                
                daily_rewards.append(hrl_reward_tensor.mean().item())
                hrl_manager.rewards.append(hrl_reward_tensor) # Append the tensor
                
                # Update Manager
                freq = opts.get('mrl_step', 10)
                if len(hrl_manager.rewards) >= freq:
                    loss = hrl_manager.update(
                        lr=opts.get('mrl_lr', 3e-4),
                        ppo_epochs=opts.get('hrl_epochs', 4),
                        clip_eps=opts.get('hrl_clip_eps', 0.2)
                    )
                    if loss is not None and tb_logger is not None:
                        tb_logger.log_value('hrl_manager_loss', loss, step)
                    print(f"HRL Update Day {day}: MeanReward={hrl_reward_tensor.mean().item():.4f} Loss={loss if loss else 0:.4f}")

            day += 1
            if day >= opts['epoch_start'] + opts['n_epochs']: break
            training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)
            
        log_training(loss_keys, table_df, opts)
        return model, hrl_manager

    # --- Original Weight Manager Logic ---
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

    # Initialize HRL Manager
    len_weights = len(cost_weights.keys())
    min_weights = [opts['mrl_range'][0] for _ in range(len_weights)]
    max_weights = [opts['mrl_range'][1] for _ in range(len_weights)]
    hrl_manager = HRLManager(
        initial_weights=cost_weights,
        history_length=opts.get('mrl_history', 10),
        hidden_size=opts.get('mrl_embedding_dim', 128),
        lr=opts.get('mrl_lr', 3e-4),
        device=opts['device'],
        min_weights=min_weights,
        max_weights=max_weights,
        ppo_epochs=opts.get('hrl_epochs', 4),
        clip_eps=opts.get('hrl_clip_eps', 0.2)
    )

    # Sync weights
    current_weights_dict = hrl_manager.get_current_weights()
    for k, v in current_weights_dict.items():
        cost_weights[k] = v

    # For tracking
    daily_rewards = []
    
    # Put model in train mode
    model.train()
    set_decode_type(model, "sampling")
    
    while True:
        # 1. HRL Manager selects new goals (weights) for the day
        new_weights = hrl_manager.select_action()
        for k, v in new_weights.items():
            cost_weights[k] = v
            
        # 2. Train Worker (Low-Level Policy) for the day using these weights
        # We reuse _train_single_day but don't pass a weight_optimizer because we manage it manually here
        step, log_pi, daily_loss, daily_total_samples, _ = _train_single_day(
            model, optimizer, baseline, lr_scheduler, scaler, None, 
            training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts
        )

        # 3. Calculate Reward for HRL Manager
        # Reward is negative total cost (or specific objective). 
        # Using 'total' cost from daily_loss
        if daily_total_samples > 0:
            # daily_loss['total'] is sum of costs. We want average cost as penalty.
            # aggregated metric for state: values of each component
            metrics = []
            avg_cost = 0
            # Order matters: must match hrl_manager.weight_names order
            for name in hrl_manager.weight_names:
                if name in daily_loss:
                    val = torch.stack(daily_loss[name]).sum().item() / daily_total_samples
                    metrics.append(val)
                else:
                    metrics.append(0.0)
            
            # Add TOTAL cost as the last metric
            avg_total_cost = torch.stack(daily_loss['total']).sum().item() / daily_total_samples
            metrics.append(avg_total_cost) # Last metric is total
            
            hrl_reward = -avg_total_cost * 0.001 # Scale reward to reasonable range (~ -30) 
            
            # Store transition
            hrl_manager.store_transition(metrics, hrl_reward)
            daily_rewards.append(hrl_reward)
            
            # Update HRL Manager
            update_freq = opts.get('mrl_step', 10)
            if len(hrl_manager.rewards) >= update_freq:
                loss = hrl_manager.update()
                if loss is not None and tb_logger is not None:
                    tb_logger.log_value('hrl_manager_loss', loss, step)
                print(f"HRL Update Day {day}: Reward={hrl_reward:.4f} Loss={loss if loss else 0:.4f}")

        # Logging
        if tb_logger is not None:
            for k, v in new_weights.items():
                tb_logger.log_value(f'hrl_weight_{k}', v, day)
        
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)

    log_training(loss_keys, table_df, opts)
    return model, hrl_manager


def train_reinforce_over_time(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    day = opts['epoch_start']
    step, training_dataset, loss_keys, table_df, args = prepare_time_dataset(optimizer, day, problem, tb_logger, cost_weights, opts)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    while True:
        step, log_pi, _, _, _ = _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, None, training_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts)
        day += 1
        if day >= opts['epoch_start'] + opts['n_epochs']: break
        training_dataset = update_time_dataset(model, optimizer, training_dataset, log_pi, day, opts, args)
    
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
        _, c_dict, l_dict = train_batch_reinforce(model, optimizer, baseline, scaler, epoch, batch_id, step, batch, tb_logger, cost_weights, opts)
        step += 1
        for key, val in zip(list(c_dict.keys()) + list(l_dict.keys()), list(c_dict.values()) + list(l_dict.values())):
            if isinstance(val, torch.Tensor): epoch_loss[key].append(val.detach()) 

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
        cost, log_likelihood, c_dict, pi = model(x, cost_weights=cost_weights, return_pi=opts['train_time'], pad=opts['train_time'], mask=mask)

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

        # Calculate loss (and normalize for gradient accumulation if accumulation_steps > 1)
        # For inactive indices: LogLikelihood is 0. So (Cost-BL)*0 = 0. Loss contribution is 0. Correct.
        reinforce_loss = (cost - bl_val) * log_likelihood
        loss = reinforce_loss.mean() + bl_loss
        loss = loss / opts['accumulation_steps']
    except Exception as e:
        if scaler is not None: autocast_context.__exit__(None, None, None)
        print(e)
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
            scaler.unscale_(optimizer)
    elif scaler is not None:
        autocast_context.__exit__(None, None, None)

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts['max_grad_norm'])
    
    # Perform optimization step with acculated gradients
    if step % opts['accumulation_steps'] == 0:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad()

    # Logging
    l_dict = {'nll': -log_likelihood, 'reinforce_loss': reinforce_loss, 'baseline_loss': bl_loss}
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
    return pi, c_dict, l_dict
