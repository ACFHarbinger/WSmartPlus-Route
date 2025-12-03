import os
import sys
import time
import torch
import pandas as pd

from tqdm import tqdm
from .meta_reinforce import (
    CostWeightManager, RewardWeightOptimizer,
    WeightContextualBandit, MORLWeightOptimizer, 
)
from src.models import WeightAdjustmentRNN
from src.utils.functions import move_to
from src.utils.log_utils import log_values, log_training, log_epoch, get_loss_stats
from .post_processing import post_processing_optimization
from .epoch import (
    set_decode_type, clip_grad_norms, 
    complete_train_pass, update_time_dataset,
    prepare_epoch, prepare_batch, prepare_time_dataset, 
)


def _train_single_day(model, optimizer, baseline, lr_scheduler, scaler, weight_optimizer, day_dataset, val_dataset, tb_logger, day, step, cost_weights, loss_keys, table_df, opts):
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
            if isinstance(val, torch.Tensor): daily_loss[key].append(val.detach()) 
            #else daily_loss[key].append(torch.tensor([val], dtype=torch.float32))
    day_duration = time.time() - start_time
    table_df.loc[day] = get_loss_stats(daily_loss)
    log_epoch(('day', day), loss_keys, daily_loss, opts)      
    _ = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, 
                            day, step, day_duration, tb_logger, cost_weights, opts)
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
        hidden_size=opts.get('rwa_embedding_dim', 64),
        lr=opts.get('mrl_lr', 0.001),
        device=opts['device'],
        meta_batch_size=opts.get('rwa_batch_size', 8),
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
    return


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
                - sum([current_weights[ckey] * (daily_cost_components[ckey] / daily_total_samples)] for ckey in cost_keys)
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
    return


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
    return


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
    return


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
    return


def train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, scaler, epoch, val_dataset, problem, tb_logger, cost_weights, opts):
    step, training_dataset, loss_keys = prepare_epoch(optimizer, epoch, problem, tb_logger, opts)
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
            if isinstance(val, torch.Tensor): epoch_loss[key].append(val.detach()) 

    epoch_duration = time.time() - start_time
    log_epoch(('epoch', epoch), loss_keys, epoch_loss, opts)
    _ = complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, 
                            epoch, step, epoch_duration, tb_logger, cost_weights, opts)    
    return


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
        cost, log_likelihood, c_dict, pi = model(x, cost_weights, return_pi=opts['train_time'], pad=opts['train_time'])

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss (and normalize for gradient accumulation if accumulation_steps > 1)
        reinforce_loss = (cost - bl_val) * log_likelihood
        loss = reinforce_loss.mean() + bl_loss
        loss = loss / opts['accumulation_steps']
    except Exception as e:
        if scaler is not None: autocast_context.__exit__(None, None, None)
        print(e)
        sys.exit(1)

    if scaler is None:
        # Perform backward pass
        loss.backward()
    else:
        # Exit autocast context
        autocast_context.__exit__(None, None, None)
        # Perform backward pass and unscale gradients
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

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
            if step % opts.get('rwa_step', 100) == 0:
                meta_loss = weight_optimizer.meta_learning_step()
                if meta_loss is not None and tb_logger is not None:
                    tb_logger.add_scalar('meta_loss', meta_loss, step)
                    
            
            # Update weights periodically
            if step % opts.get('rwa_update_step', 500) == 0:
                updated = weight_optimizer.update_weights()
    
            if step % opts.get('rwa_step', 100) == 0 or step % opts.get('rwa_update_step', 500) == 0:
                if tb_logger is not None and (meta_loss is not None or updated is not None):
                    for i, name in enumerate(weight_optimizer.weight_names):
                        tb_logger.add_scalar(f'weight_{name}', weight_optimizer.current_weights[i].item(), step)
    return pi, c_dict, l_dict
