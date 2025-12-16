
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
                 hidden_dim=128,
                 lstm_hidden=64,
                 num_layers_gat=3,
                 num_heads=8,
                 dropout=0.1,
                 device='cuda'):
        super(GATLSTManager, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.input_dim_dynamic = input_dim_dynamic
        
        # 1. Temporal Encoder (LSTM)
        # Inputs: (Batch * N, Sequence, 1) - We treat each node's history as a sequence
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
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
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Route Gate Head (Actor 2)
        # Global Pooling -> MLP -> (Batch, 2) -> Logits for [No Route, Route]
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Critic (Value Function)
        # Global Pooling -> MLP -> (Batch, 1)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.to(device)
        
        # Optimization
        # We will set optimizer externally or manually
        self.optimizer = None 

        # Buffers for PPO
        self.clear_memory()

    def clear_memory(self):
        self.states_static = []
        self.states_dynamic = []
        self.actions_mask = []
        self.actions_gate = []
        self.log_probs_mask = []
        self.log_probs_gate = []
        self.rewards = []
        self.values = []
        self.dones = []

    def feature_processing(self, static, dynamic):
        """
        static: (Batch, N, 2)
        dynamic: (Batch, N, History)
        Returns: (Batch, N, Hidden)
        """
        B, N, H_len = dynamic.size()
        
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        dyn_flat = dynamic.view(B*N, H_len, 1)
        
        # Run LSTM
        # out: (Batch*N, History, Hidden), (h_n, c_n)
        _, (h_n, _) = self.lstm(dyn_flat)
        # h_n: (1, Batch*N, Hidden) -> (Batch, N, Hidden)
        temporal_embed = h_n.squeeze(0).view(B, N, -1)
        
        # Concatenate with static
        # static: (Batch, N, 2)
        combined = torch.cat([static, temporal_embed], dim=2)
        
        # Project to hidden
        x = self.feature_embedding(combined)
        
        return x

    def forward(self, static, dynamic):
        """
        Returns logits for mask and gate, and state value.
        """
        # Encode
        x = self.feature_processing(static, dynamic) # (B, N, H)
        
        # Spatial Context
        # Transformer expects (Batch, Seq, F) with batch_first=True
        h = self.gat_encoder(x) # (B, N, H)
        
        # Node Mask Logits
        mask_logits = self.mask_head(h) # (B, N, 2)
        
        # Global Pooling for Gate/Critic
        # Mean pooling
        h_global = torch.mean(h, dim=1) # (B, H)
        
        # Gate Logits
        gate_logits = self.gate_head(h_global) # (B, 2)
        
        # Value
        value = self.critic(h_global) # (B, 1)
        
        return mask_logits, gate_logits, value

    def select_action(self, static, dynamic, deterministic=False):
        """
        Sample actions from policy.
        """
        mask_logits, gate_logits, value = self.forward(static, dynamic)
        
        # 1. Gate Decision
        gate_probs = F.softmax(gate_logits, dim=-1)
        if deterministic:
            gate_action = torch.argmax(gate_probs, dim=-1)
        else:
            gate_dist = torch.distributions.Categorical(gate_probs)
            gate_action = gate_dist.sample()
            gate_log_prob = gate_dist.log_prob(gate_action)
            
        # 2. Node Mask Decision
        # (Batch, N, 2)
        mask_probs = F.softmax(mask_logits, dim=-1)
        if deterministic:
            mask_action = torch.argmax(mask_probs, dim=-1)
        else:
            mask_dist = torch.distributions.Categorical(mask_probs)
            mask_action = mask_dist.sample() # (Batch, N)
            mask_log_prob = mask_dist.log_prob(mask_action).sum(dim=1) # Sum log probs over nodes? 
            # Yes, standard for joint action over independent nodes assumption
            
        # Store for PPO
        if not deterministic:
            # Mask Log Prob: Sum over nodes
            # Gate Log Prob: Single scalar
            log_prob_mask = mask_dist.log_prob(mask_action).sum(dim=1)
            log_prob_gate = gate_dist.log_prob(gate_action)
            
            self.states_static.append(static.detach().cpu())
            self.states_dynamic.append(dynamic.detach().cpu())
            self.actions_gate.append(gate_action.detach().cpu())
            self.actions_mask.append(mask_action.detach().cpu())
            self.log_probs_gate.append(log_prob_gate.detach().cpu())
            self.log_probs_mask.append(log_prob_mask.detach().cpu())
            self.values.append(value.detach().cpu())
        
        return mask_action, gate_action, value

    def update(self, lr=3e-4, gamma=0.99, clip_eps=0.2, ppo_epochs=4):
        """
        PPO Update with combined loss for Gate and Mask.
        """
        if not self.rewards:
            return None
            
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Convert buffers to tensors (Keep on CPU initially to save VRAM)
        # Flattening (Step, Batch, ...) -> (TotalSamples, ...)
        old_states_static = torch.cat(self.states_static)
        old_states_dynamic = torch.cat(self.states_dynamic)
        old_mask_actions = torch.cat(self.actions_mask)
        old_gate_actions = torch.cat(self.actions_gate)
        old_log_probs_mask = torch.cat(self.log_probs_mask)
        old_log_probs_gate = torch.cat(self.log_probs_gate)
        old_values = torch.cat(self.values).squeeze(-1)
        
        # Rewards processing
        sample = self.rewards[0]
        if isinstance(sample, (int, float)) or (torch.is_tensor(sample) and sample.ndim == 0):
             # Expand scalar reward to batch size
             batch_size = old_states_static.shape[0] // len(self.rewards)
             expanded_rewards = []
             for r in self.rewards:
                 r_val = r if isinstance(r, (int, float)) else r.item()
                 # Use CPU tensor for expansion
                 expanded_rewards.append(torch.full((batch_size,), r_val))
             rewards = torch.cat(expanded_rewards)
        else:
            rewards = torch.cat(self.rewards).cpu() # Ensure CPU
            
        # Calculate Returns & Advantages (on CPU)
        returns = rewards
        advantages = returns - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages - advantages.mean())
            
        # Mini-batch PPO
        total_samples = old_states_static.size(0)
        batch_size = 1024 # Mini-batch size
        
        total_loss = 0
        updates = 0
        
        for _ in range(ppo_epochs):
            indices = torch.randperm(total_samples)
            
            for i in range(0, total_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                
                # Move batch to device
                b_static = old_states_static[batch_idx].to(self.device)
                b_dynamic = old_states_dynamic[batch_idx].to(self.device)
                b_mask_act = old_mask_actions[batch_idx].to(self.device)
                b_gate_act = old_gate_actions[batch_idx].to(self.device)
                b_old_log_mask = old_log_probs_mask[batch_idx].to(self.device)
                b_old_log_gate = old_log_probs_gate[batch_idx].to(self.device)
                b_returns = returns[batch_idx].to(self.device)
                b_adv = advantages[batch_idx].to(self.device)
                
                # Forward pass
                mask_logits, gate_logits, values = self.forward(b_static, b_dynamic)
                values = values.squeeze(-1)
                
                # New Log Probs
                mask_dist = torch.distributions.Categorical(logits=mask_logits)
                gate_dist = torch.distributions.Categorical(logits=gate_logits)
                
                new_log_probs_mask = mask_dist.log_prob(b_mask_act).sum(dim=1)
                new_log_probs_gate = gate_dist.log_prob(b_gate_act)
                
                new_log_probs = new_log_probs_mask + new_log_probs_gate
                old_log_probs = b_old_log_mask + b_old_log_gate
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, b_returns)
                
                entropy = mask_dist.entropy().mean() + gate_dist.entropy().mean()
                
                loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                updates += 1
            
        self.clear_memory()
        return total_loss / updates if updates > 0 else 0
