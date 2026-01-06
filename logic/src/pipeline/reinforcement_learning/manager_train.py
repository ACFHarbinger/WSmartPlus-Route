import torch
import torch.nn.functional as F

def update_manager_ppo(manager, optimizer=None, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4, lambda_mask_aux=0.0, entropy_coef=0.1):
    """
    PPO Update with combined loss for Gate and Mask.
    Now uses proper temporal discounted returns.
    Moves training logic out of the model class.
    """
    if not manager.rewards:
        return None
        
    if optimizer is None:
        optimizer = torch.optim.Adam(manager.parameters(), lr=lr)

    # Convert buffers to tensors 
    # T: Time steps (days), B: Batch size (instances)
    # manager.rewards is a list of T tensors, each of shape (B,)
    
    # Calculate Returns & Advantages (on CPU)
    # Returns G_t = R_t + gamma * G_{t+1}
    T = len(manager.rewards)
    B = manager.rewards[0].size(0)
    
    rewards_tensor = torch.stack(manager.rewards).cpu() # (T, B)
    returns_tensor = torch.zeros_like(rewards_tensor) # (T, B)
    
    # Compute returns backwards
    running_return = torch.zeros(B)
    for t in reversed(range(T)):
        running_return = rewards_tensor[t] + gamma * running_return
        returns_tensor[t] = running_return
        
    # Flatten all buffers for PPO (TotalSamples = T * B)
    returns = returns_tensor.flatten() # (T*B,)
    
    old_states_static = torch.cat(manager.states_static)  # (T*B, N, 2)
    old_states_dynamic = torch.cat(manager.states_dynamic) # (T*B, N, H)
    old_states_global = torch.cat(manager.states_global)   # (T*B, G)
    old_mask_actions = torch.cat(manager.actions_mask)     # (T*B, N)
    old_gate_actions = torch.cat(manager.actions_gate)     # (T*B,)
    old_log_probs_mask = torch.cat(manager.log_probs_mask) # (T*B,)
    old_log_probs_gate = torch.cat(manager.log_probs_gate) # (T*B,)
    old_values = torch.cat(manager.values).squeeze(-1)    # (T*B,)
    old_target_masks = torch.cat(manager.target_masks)    # (T*B, N)
    
    # Advantage Calculation
    advantages = returns - old_values
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        advantages = (advantages - advantages.mean())
        
    # Mini-batch PPO
    states_size = old_states_static.size(0)
    total_loss = 0
    updates = 0
    for _ in range(ppo_epochs):
        indices = torch.randperm(states_size)
        
        for i in range(0, states_size, manager.batch_size):
            batch_idx = indices[i:i+manager.batch_size]
            
            # Move batch to device
            b_static = old_states_static[batch_idx].to(manager.device)
            b_dynamic = old_states_dynamic[batch_idx].to(manager.device)
            b_global = old_states_global[batch_idx].to(manager.device) 
            b_mask_act = old_mask_actions[batch_idx].to(manager.device)
            b_gate_act = old_gate_actions[batch_idx].to(manager.device)
            b_old_log_mask = old_log_probs_mask[batch_idx].to(manager.device)
            b_old_log_gate = old_log_probs_gate[batch_idx].to(manager.device)
            b_returns = returns[batch_idx].to(manager.device)
            b_adv = advantages[batch_idx].to(manager.device)
            b_target_mask = old_target_masks[batch_idx].to(manager.device)
            
            # Forward pass
            mask_logits, gate_logits, values = manager.forward(b_static, b_dynamic, b_global)
            values = values.squeeze(-1)
            
            # New Log Probs
            mask_dist = torch.distributions.Categorical(logits=mask_logits)
            gate_dist = torch.distributions.Categorical(logits=gate_logits)
            
            new_log_probs_mask = mask_dist.log_prob(b_mask_act).sum(dim=1)
            new_log_probs_gate = gate_dist.log_prob(b_gate_act)
            
            new_log_probs = new_log_probs_mask + new_log_probs_gate
            old_log_probs = b_old_log_mask + b_old_log_gate
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # HICRA-Inspired Credit Assignment
            # Weight gradient updates by overflow severity
            b_overflow = (b_dynamic[:, :, -1] > 0.9).float().sum(dim=1)  # Count of critical nodes
            credit_weight = 1.0 + (b_overflow * 0.5)  # Higher weight for instances with more critical nodes
            credit_weight = credit_weight / credit_weight.mean()  # Normalize
            
            surr1 = ratio * b_adv * credit_weight
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv * credit_weight
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, b_returns)
            
            entropy = mask_dist.entropy().mean() + gate_dist.entropy().mean()
            
            # Use target_mask provided during select_action (which mimics VRPP)
            target_mask = b_target_mask 
            
            # Correct Logit for BCE: s1 - s0 (Keep - Skip)
            logits_diff = mask_logits[:, :, 1] - mask_logits[:, :, 0]
            
            # Pressure Mirroring
            # Intensify Expert Signal
            pos_weight = torch.tensor([50.0]).to(manager.device)
            loss_mask_aux = F.binary_cross_entropy_with_logits(logits_diff, target_mask, pos_weight=pos_weight)

            loss = actor_loss + 0.5 * value_loss - entropy_coef * entropy + (lambda_mask_aux * loss_mask_aux)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.parameters(), 1.0) # Clip gradients to prevent NaN
            optimizer.step()
            
            total_loss += loss.item()
            updates += 1
        
    manager.clear_memory()
    return total_loss / updates if updates > 0 else 0
