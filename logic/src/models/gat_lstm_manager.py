
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLSTManager(nn.Module):
    """
    Hierarchical Manager Agent for PCVRP.
    Uses LSTM to encode temporal waste patterns and GAT (Transformer) to encode spatial structure.
    Outputs:
        1. Node Mask: Which nodes to visit today.
        2. Route Gate: Whether to dispatch vehicles today.
    """
    def __init__(self,
                 input_dim_static=2,   # x, y
                 input_dim_dynamic=10, # waste history length
                 global_input_dim=5,   # avg_waste, overflows, avg_len, visited_ratio, day (or customized)
                 batch_size=1024,
                 hidden_dim=128,
                 lstm_hidden=64,
                 num_layers_gat=3,
                 num_heads=8,
                 dropout=0.1,
                 device='cuda'):
        super(GATLSTManager, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_dim_dynamic = input_dim_dynamic
        self.global_input_dim = global_input_dim
        
        # 1. Temporal Encoder (Simplified: Current Waste Projector)
        # Inputs: (Batch, N, 1) - Current Waste Only
        self.waste_proj = nn.Linear(1, lstm_hidden)
        # self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
        # 2. Feature Fusion
        # Static (2) + Temporal Embedding (lstm_hidden) -> Hidden
        self.feature_embedding = nn.Linear(input_dim_static + lstm_hidden, hidden_dim)
        
        # 3. Spatial Encoder (GAT / Transformer)
        # Using TransformerEncoder as standard "GAT" on fully connected graph
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, 
                                                 dim_feedforward=hidden_dim*4, dropout=dropout,
                                                 batch_first=True)
        self.gat_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_gat)
        
        # 4. Heads
        
        # Node Mask Head (Actor 1)
        # Output: (Batch, N, 2) -> Logits for [Skip, Visit]
        # Input: Hidden + Skip Connection (lstm_hidden)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Route Gate Head (Actor 2)
        # Global Pooling -> MLP -> (Batch, 2) -> Logits for [No Route, Route]
        # Input: Hidden + Global Features
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim + global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Critic (Value Function)
        # Global Pooling -> MLP -> (Batch, 1)
        # Input: Hidden + Global Features
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
        
        # Optimization
        # We will set optimizer externally or manually
        self.optimizer = None 

        # Buffers for PPO
        self.clear_memory()

        # Force the Manager to prefer Routing (1) over Skipping (0) at the start
        with torch.no_grad():
            # Assuming gate_head's last layer is a Linear layer
            self.gate_head[-1].bias.fill_(0)
            
            # Revert to Neutral Bias for Attempt 70
            self.mask_head[-1].bias.fill_(0)

    def clear_memory(self):
        self.states_static = []
        self.states_dynamic = []
        self.states_global = []
        self.actions_mask = []
        self.actions_gate = []
        self.log_probs_mask = []
        self.log_probs_gate = []
        self.rewards = []
        self.values = []
        self.target_masks = [] # New buffer for target masks
        self.dones = []

    def feature_processing(self, static, dynamic):
        """
        static: (Batch, N, 2)
        dynamic: (Batch, N, History)
        Returns: (Batch, N, Hidden)
        """
        B, N, H_len = dynamic.size()
        
        # Simplify: Use Current Waste (Last Step)
        # dynamic: (Batch, N, History)
        current_waste = dynamic[:, :, -1:] # (Batch, N, 1)
        
        # Project
        temporal_embed = self.waste_proj(current_waste) # (Batch, N, lstm_hidden)
        
        # Original LSTM Logic (Commented out)
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        # dyn_flat = dynamic.view(B*N, H_len, 1)
        # _, (h_n, _) = self.lstm(dyn_flat)
        # temporal_embed = h_n.squeeze(0).view(B, N, -1)
        
        # Concatenate with static
        # static: (Batch, N, 2)
        combined = torch.cat([static, temporal_embed], dim=2)
        
        # Project to hidden
        x = self.feature_embedding(combined)
        
        return x

    def forward(self, static, dynamic, global_features):
        """
        Returns logits for mask and gate, and state value.
        global_features: (Batch, global_input_dim)
        """
        # Encode
        x = self.feature_processing(static, dynamic) # (B, N, H)
        
        # Spatial Context
        # Transformer expects (Batch, Seq, F) with batch_first=True
        h = self.gat_encoder(x) # (B, N, H)
        
        # SKIP CONNECTION: Concatenate temporal_embed (Waste Proj) to h
        # Recalculate temporal_embed or return it from feature_processing?
        # feature_processing assumes it's internal.
        # Let's refactor feature_processing to return temporal_embed as well.
        # OR just recompute it here since it's cheap (Linear).
        current_waste = dynamic[:, :, -1:]
        temporal_embed = self.waste_proj(current_waste)
        
        # h_skip: (B, N, H + lstm_hidden)
        h_skip = torch.cat([h, temporal_embed], dim=-1)
        
        # Node Mask Logits
        mask_logits = self.mask_head(h_skip) # (B, N, 2)
        
        # Global Pooling for Gate/Critic
        # Combine Mean (for general load) and Max (for urgency/overflows)
        h_mean = torch.mean(h, dim=1) # (B, H)
        h_max = torch.max(h, dim=1)[0] # (B, H)
        h_global = h_mean + h_max      # Simple addition or concatenation
        
        # Concatenate Global Features
        # global_features: (B, G)
        h_combined = torch.cat([h_global, global_features], dim=1) # (B, H+G)

        # Gate Logits
        gate_logits = self.gate_head(h_combined) # (B, 2)
        
        # Value
        value = self.critic(h_combined) # (B, 1)
        
        return mask_logits, gate_logits, value

    def select_action(self, static, dynamic, global_features=None, deterministic=False, force_action=None, force_node_mask=None, threshold=0.5, mask_threshold=0.5, target_mask=None):
        """
        static: (Batch, N, 2) - Locations
        dynamic: (Batch, N, 1) - Waste levels
        global_features: (Batch, 3) - Global context
        force_action: (Batch,) of 0/1 or None - Expert action to force (Gate)
        force_node_mask: (Batch, N) of 0/1 or None - Expert action to force (Mask)
        threshold: float - Probability threshold for gate=1 (Route) when deterministic
        mask_threshold: float - Probability threshold for mask=1 (Unmask) when deterministic
        target_mask: (Batch, N) of 0/1 or None - Expert target for mask auxiliary loss
        """
        mask_logits, gate_logits, value = self.forward(static, dynamic, global_features)
        
        # 1. Gate Decision (0=Skip, 1=Route)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = torch.distributions.Categorical(gate_probs)
        
        if force_action is not None and torch.is_tensor(force_action):
            sampled_action = gate_dist.sample()
            force_mask = (force_action != -1)
            gate_action = torch.where(force_mask, force_action, sampled_action)
        elif deterministic:
            if threshold < 0:
                # Force Gate Open
                gate_action = torch.ones_like(gate_probs[..., 1]).long()
            else:
                gate_action = (gate_probs[..., 1] > threshold).long()
        else:
            gate_action = gate_dist.sample()
            
        # 2. Mask Decision (Which nodes to mask/unmask)
        mask_probs = F.softmax(mask_logits, dim=-1)
        if deterministic:
            mask_action = (mask_probs[..., 1] > mask_threshold).long()
        else:
            mask_dist = torch.distributions.Categorical(mask_probs)
            mask_action = mask_dist.sample() # (Batch, N)
            
        # Apply safety railing (Inference Only)
        if deterministic:
            curr_waste = dynamic[:, :, -1] 
            safety_mask = (curr_waste > 0.85).long() # Proactive unmasking
            mask_action = torch.max(mask_action, safety_mask)
            
        # Apply force_node_mask if provided (Expert Forcing - Training Only)
        if force_node_mask is not None:
            mask_action = torch.max(mask_action, force_node_mask.long())
            
        # Store for PPO
        if not deterministic:
            log_prob_mask = mask_dist.log_prob(mask_action).sum(dim=1)
            log_prob_gate = gate_dist.log_prob(gate_action)
            
            self.states_static.append(static.detach().cpu())
            self.states_dynamic.append(dynamic.detach().cpu())
            self.states_global.append(global_features.detach().cpu()) 
            self.actions_gate.append(gate_action.detach().cpu())
            self.actions_mask.append(mask_action.detach().cpu())
            self.log_probs_gate.append(log_prob_gate.detach().cpu())
            self.log_probs_mask.append(log_prob_mask.detach().cpu())
            self.values.append(value.detach().cpu())
            
            # Store target mask for auxiliary loss
            if target_mask is not None:
                self.target_masks.append(target_mask.detach().cpu())
            else:
                # Fallback to current waste > 0.9
                self.target_masks.append((dynamic[:, :, -1] > 0.9).float().detach().cpu())
        
        return mask_action, gate_action, value

    def update(self, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4, lambda_mask_aux=0.0, entropy_coef=0.1):
        """
        PPO Update with combined loss for Gate and Mask.
        Now uses proper temporal discounted returns.
        """
        if not self.rewards:
            return None
            
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert buffers to tensors 
        # T: Time steps (days), B: Batch size (instances)
        # self.rewards is a list of T tensors, each of shape (B,)
        
        # Calculate Returns & Advantages (on CPU)
        # Returns G_t = R_t + gamma * G_{t+1}
        T = len(self.rewards)
        B = self.rewards[0].size(0)
        
        rewards_tensor = torch.stack(self.rewards).cpu() # (T, B)
        returns_tensor = torch.zeros_like(rewards_tensor) # (T, B)
        
        # Compute returns backwards
        running_return = torch.zeros(B)
        for t in reversed(range(T)):
            running_return = rewards_tensor[t] + gamma * running_return
            returns_tensor[t] = running_return
            
        # Flatten all buffers for PPO (TotalSamples = T * B)
        returns = returns_tensor.flatten() # (T*B,)
        
        old_states_static = torch.cat(self.states_static)  # (T*B, N, 2)
        old_states_dynamic = torch.cat(self.states_dynamic) # (T*B, N, H)
        old_states_global = torch.cat(self.states_global)   # (T*B, G)
        old_mask_actions = torch.cat(self.actions_mask)     # (T*B, N)
        old_gate_actions = torch.cat(self.actions_gate)     # (T*B,)
        old_log_probs_mask = torch.cat(self.log_probs_mask) # (T*B,)
        old_log_probs_gate = torch.cat(self.log_probs_gate) # (T*B,)
        old_values = torch.cat(self.values).squeeze(-1)    # (T*B,)
        old_target_masks = torch.cat(self.target_masks)    # (T*B, N)
        
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
            
            for i in range(0, states_size, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Move batch to device
                b_static = old_states_static[batch_idx].to(self.device)
                b_dynamic = old_states_dynamic[batch_idx].to(self.device)
                b_global = old_states_global[batch_idx].to(self.device) # New
                b_mask_act = old_mask_actions[batch_idx].to(self.device)
                b_gate_act = old_gate_actions[batch_idx].to(self.device)
                b_old_log_mask = old_log_probs_mask[batch_idx].to(self.device)
                b_old_log_gate = old_log_probs_gate[batch_idx].to(self.device)
                b_returns = returns[batch_idx].to(self.device)
                b_adv = advantages[batch_idx].to(self.device)
                b_target_mask = old_target_masks[batch_idx].to(self.device)
                
                # Forward pass
                mask_logits, gate_logits, values = self.forward(b_static, b_dynamic, b_global)
                values = values.squeeze(-1)
                
                # New Log Probs
                mask_dist = torch.distributions.Categorical(logits=mask_logits)
                gate_dist = torch.distributions.Categorical(logits=gate_logits)
                
                new_log_probs_mask = mask_dist.log_prob(b_mask_act).sum(dim=1)
                new_log_probs_gate = gate_dist.log_prob(b_gate_act)
                
                new_log_probs = new_log_probs_mask + new_log_probs_gate
                old_log_probs = b_old_log_mask + b_old_log_gate
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Attempt 62: HICRA-Inspired Credit Assignment
                # Weight gradient updates by overflow severity
                # This focuses learning on high-impact decisions
                b_overflow = (b_dynamic[:, :, -1] > 0.9).float().sum(dim=1)  # Count of critical nodes
                credit_weight = 1.0 + (b_overflow * 0.5)  # Higher weight for instances with more critical nodes
                credit_weight = credit_weight / credit_weight.mean()  # Normalize
                
                surr1 = ratio * b_adv * credit_weight
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv * credit_weight
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, b_returns)
                
                entropy = mask_dist.entropy().mean() + gate_dist.entropy().mean()
                
                # Attempt 67: Precision Lookahead Masking
                # Use target_mask provided during select_action (which mimics VRPP)
                target_mask = b_target_mask 
                
                # Correct Logit for BCE: s1 - s0 (Keep - Skip)
                logits_diff = mask_logits[:, :, 1] - mask_logits[:, :, 0]
                
                # Attempt 71: Pressure Mirroring
                # Attempt 73: Intensify Expert Signal
                pos_weight = torch.tensor([50.0]).to(self.device)
                loss_mask_aux = F.binary_cross_entropy_with_logits(logits_diff, target_mask, pos_weight=pos_weight)
                
                # Check Stats (Logging)
                if _ == 0 and i == 0:
                     mask_probs = torch.sigmoid(logits_diff) 
                     mask_full = (target_mask == 1.0)
                     mask_empty = (target_mask == 0.0)
                     
                     p_full = mask_probs[mask_full].mean().item() if mask_full.any() else 0.0
                     p_empty = mask_probs[mask_empty].mean().item() if mask_empty.any() else 0.0
                     print(f" Mask Aux Stat (Attempt 73): Full P(Keep)={p_full:.4f} | Empty P(Keep)={p_empty:.4f} | AuxLoss={loss_mask_aux.item():.4f}")

                loss = actor_loss + 0.5 * value_loss - entropy_coef * entropy + (lambda_mask_aux * loss_mask_aux)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # Clip gradients to prevent NaN
                self.optimizer.step()
                
                total_loss += loss.item()
                updates += 1
            
        self.clear_memory()
        return total_loss / updates if updates > 0 else 0
