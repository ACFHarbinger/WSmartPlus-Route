import os
import time
import torch
import numpy as np

from tqdm import tqdm
from collections import deque
from backend.src.utils.functions import move_to
from backend.src.utils.log_utils import log_values
from .epoch import (
    validate, complete_train_pass,
    clip_grad_norms, set_decode_type,
    prepare_epoch, update_time_dataset, 
)


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Tensor of shape [batch_size]
        values: Tensor of shape [batch_size]
        next_values: Tensor of shape [batch_size] or scalar for last value
        dones: Tensor of shape [batch_size] indicating episode termination
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: Tensor of shape [batch_size]
        returns: Tensor of shape [batch_size]
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
        
    returns = advantages + values
    return advantages, returns


def train_ppo_over_time(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    step, is_cuda, training_dataset = prepare_epoch(optimizer, 0, problem, tb_logger, opts)

    # Multi-epoch tracking
    val_history = deque(maxlen=opts.get('val_window', 5))  # Track last 5 days' validation costs
    best_val_cost = float('inf')
    patience = opts.get('patience', 10)
    patience_counter = 0
    
    # PPO hyperparameters
    clip_eps = opts.get('clip_eps', 0.2)
    vf_coef = opts.get('vf_coef', 0.5)
    entropy_coef = opts.get('entropy_coef', 0.01) 
    ppo_epochs = opts.get('k_ppo_epochs', 4)
    mini_batch_size = opts.get('mini_batch_size', 64)
    gamma = opts.get('disc_gamma', 0.99)
    lam = opts.get('gae_lambda', 0.95)
    target_kl = opts.get('target_kl', 0.015)
    
    # Create separate optimizer for baseline if not already using one
    baseline_optimizer = optimizer if not opts.get('separate_value_opt', False) else torch.optim.RMSprop(
        baseline.parameters(), lr=opts.get('lr_critic_value', 3e-4))
    value_updates = opts.get('value_updates', 1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    for day in range(opts['epoch_start'], opts['epoch_start'] + opts['n_epochs']):
        log_pi = []
        log_probs = []
        training_dataloader = torch.utils.data.DataLoader(
            baseline.wrap_dataset(training_dataset), batch_size=opts['batch_size'], pin_memory=True)
        start_time = time.time()
        if scaler is None:
            for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts['no_progress_bar'])):
                pi, log_prob = train_batch_ppo_extra(
                    model, optimizer, baseline, baseline_optimizer, day, batch_id, step, batch, 
                    tb_logger, cost_weights, clip_eps, vf_coef, entropy_coef, ppo_epochs, 
                    mini_batch_size, gamma, lam, target_kl, value_updates, opts
                )
                log_pi.append(pi)
                log_probs.append(log_prob)
                step += 1

        epoch_duration = time.time() - start_time
        
        # Validation and multi-epoch adjustments
        if opts['val_size'] > 0:
            val_cost = validate(model, val_dataset, opts)
            val_history.append(val_cost.item())
            if not opts['no_tensorboard']:
                tb_logger.log_value('val_avg_cost', val_cost, step)
            
            if len(val_history) == val_history.maxlen:
                # Analyze trend
                slope = np.polyfit(range(len(val_history)), list(val_history), 1)[0]
                if slope > -0.001:  # Validation cost not improving significantly
                    clip_eps = max(0.05, clip_eps * 0.9)
                    print(f"Day {day}: Adjusted clip_eps to {clip_eps}")
                    # Optionally reduce LR
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95
                    
                    if not opts['no_tensorboard']:
                        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
            
            # Early stopping
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(opts['save_dir'], 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at day {day}")
                    break
        
        complete_train_pass(model, optimizer, baseline, lr_scheduler, val_dataset, day, step, epoch_duration, tb_logger, is_cuda, opts)
        log_pi = torch.stack(log_pi).contiguous().view(-1, log_pi[0].size(1))
        training_dataset = update_time_dataset(optimizer, training_dataset, log_pi, day, opts)
        if opts['val_size'] > 0:
            model.load_state_dict(torch.load(os.path.join(opts['save_dir'], 'best_model.pth')))
    return


def train_batch_ppo_extra(model, optimizer, baseline, baseline_optimizer, epoch, batch_id, step, 
                            batch, tb_logger, cost_weights, clip_eps, vf_coef, entropy_coef, 
                            ppo_epochs, mini_batch_size, gamma, lam, target_kl, value_updates, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts['device'])
    bl_val = move_to(bl_val, opts['device']) if bl_val is not None else None
    batch_size = x['loc'].size(0)
    
    # Get initial policy distribution and costs before any updates
    with torch.no_grad():
        cost, old_log_likelihood, old_pi = model(x, cost_weights, return_pi=opts['train_time'], pad=opts['train_time'])
        old_values = baseline.eval(x, detach_cost=True)[0] if bl_val is None else bl_val
        
    # If we need to compute GAE, we need to get next state values
    # For simplicity, use current state values as approximation
    with torch.no_grad():
        next_values = old_values.clone()
        
        # For terminal states, next_value should be 0
        # Assuming we have a 'done' flag in x, otherwise we can approximate or omit this
        dones = torch.zeros(batch_size, device=opts['device'])
        if 'done' in x: dones = x['done']
            
        # Compute GAE and returns
        advantages, returns = compute_gae(cost, old_values, next_values, dones, gamma, lam)
        if opts['normalize_adv']: advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Store old policy and value for optimization
    old_policy = {
        'log_likelihood': old_log_likelihood.detach(),
        'values': old_values.detach()
    }
    
    # Multiple PPO epochs over the same batch
    for ppo_epoch in range(ppo_epochs):
        # Process data in mini-batches for better stability
        indices = torch.randperm(batch_size)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        for start_idx in range(0, batch_size, mini_batch_size):
            idx = indices[start_idx:start_idx + mini_batch_size]
            
            # Get mini-batch data
            mini_batch_x = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            mini_batch_advantages = advantages[idx]
            mini_batch_returns = returns[idx]
            mini_batch_old_log_likelihood = old_policy['log_likelihood'][idx]
            mini_batch_old_values = old_policy['values'][idx]
            
            # Evaluate model with current policy
            mb_cost, mb_log_likelihood, mb_pi = model(
                mini_batch_x, cost_weights, return_pi=opts['train_time'], pad=opts['train_time']
            )
            
            # Policy ratio and clipped objective
            ratio = torch.exp(mb_log_likelihood - mini_batch_old_log_likelihood)
            surr1 = ratio * mini_batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mini_batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            new_values, _ = baseline.eval(mini_batch_x, mb_cost)
            
            # Clipped value function objective
            values_clipped = mini_batch_old_values + torch.clamp(
                new_values - mini_batch_old_values, -clip_eps, clip_eps
            )
            vf_loss1 = (new_values - mini_batch_returns).pow(2)
            vf_loss2 = (values_clipped - mini_batch_returns).pow(2)
            value_loss = torch.max(vf_loss1, vf_loss2).mean()
            
            # Entropy bonus
            policy_entropy = -(torch.exp(mb_log_likelihood) * mb_log_likelihood).sum(dim=-1).mean()
            
            # KL divergence for early stopping
            approx_kl = (mini_batch_old_log_likelihood - mb_log_likelihood).mean()
            total_kl += approx_kl.item()
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss - entropy_coef * policy_entropy
            
            # Normalize loss for gradient accumulation
            loss = loss / opts['accumulation_steps']
            
            # Policy network update (for actor)
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradient norms
            grad_norms = clip_grad_norms(optimizer.param_groups, opts['max_grad_norm'])
            
            # Optimization step for policy
            if step % opts['accumulation_steps'] == 0:
                optimizer.step()
                
            # Separate value network updates (if using a separate optimizer)
            if opts.get('separate_value_opt', False):
                for _ in range(value_updates):
                    baseline_optimizer.zero_grad()
                    new_values, _ = baseline.eval(mini_batch_x, mb_cost.detach())
                    value_loss = (new_values - mini_batch_returns).pow(2).mean()
                    value_loss.backward()
                    baseline_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += policy_entropy.item()
        
        # Calculate average KL divergence across all mini-batches
        avg_kl = total_kl / ((batch_size + mini_batch_size - 1) // mini_batch_size)
        
        # Early stopping based on KL divergence
        if avg_kl > target_kl:
            print(f"Early stopping at ppo_epoch {ppo_epoch} due to reaching KL divergence threshold ({avg_kl:.4f} > {target_kl:.4f})")
            break
        
        # Dynamic clip parameter adjustment based on KL divergence
        if avg_kl > target_kl * 1.5:
            clip_eps *= 0.8
        elif avg_kl < target_kl * 0.5:
            clip_eps *= 1.2
        clip_eps = min(max(clip_eps, 0.05), 0.3)  # Keep within reasonable bounds
    
    # Logging
    if step % opts['log_step'] == 0:
        # Using original log_values function but adding our new metrics
        log_values(cost, grad_norms, epoch, batch_id, step, old_log_likelihood, 
                  total_policy_loss / ppo_epochs, total_value_loss / ppo_epochs, tb_logger, opts)

        # Log additional PPO metrics
        if not opts['no_tensorboard']:
            tb_logger.log_value('entropy', total_entropy / ppo_epochs, step)
            tb_logger.log_value('kl_divergence', avg_kl, step)
            tb_logger.log_value('clip_eps', clip_eps, step)
    
    return old_pi, old_log_likelihood  # Return for consistency with original API


def train_batch_ppo_simple(model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, cost_weights, clip_eps, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts['device'])
    bl_val = move_to(bl_val, opts['device']) if bl_val is not None else None

    # Compute old policy log probabilities (before update)
    with torch.no_grad():
        _, old_log_likelihood, old_pi = model(x, cost_weights, return_pi=opts['train_time'], pad=opts['train_time'])
    
    # Evaluate model with current policy
    cost, log_likelihood, pi = model(x, cost_weights, return_pi=opts['train_time'], pad=opts['train_time'])

    # Evaluate baseline
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    advantage = cost - bl_val  # Advantage estimation
    # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    # PPO clipped objective
    ratio = torch.exp(log_likelihood - old_log_likelihood)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    
    #policy_entropy = -(torch.exp(log_likelihood) * log_likelihood).sum(dim=-1).mean()
    #entropy_coef = 0.01
    #vf_coef = 0.5
    #loss = policy_loss + vf_coef * bl_loss - entropy_coef * policy_entropy
    policy_loss = -torch.min(surr1, surr2).mean()
    loss = policy_loss + bl_loss

    # Normalize loss for gradient accumulation
    loss = loss / opts['accumulation_steps']

    # Perform backward pass
    loss.backward()

    # Clip gradient norms
    grad_norms = clip_grad_norms(optimizer.param_groups, opts['max_grad_norm'])
    
    # Optimization step
    if step % opts['accumulation_steps'] == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % opts['log_step'] == 0:
        log_values(cost, grad_norms, epoch, batch_id, step, log_likelihood, policy_loss, bl_loss, tb_logger, opts)
    
    return pi, old_log_likelihood  # Return old log prob for consistency