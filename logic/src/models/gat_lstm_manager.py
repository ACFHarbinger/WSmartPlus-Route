
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
                 global_input_dim=2,   # avg_waste, max_waste (Only current)
                 critical_threshold=0.9,
                 batch_size=1024,
                 hidden_dim=128,
                 lstm_hidden=64,
                 num_layers_gat=3,
                 num_heads=8,
                 dropout=0.1,
                 device='cuda',
                 shared_encoder=None):
        super(GATLSTManager, self).__init__()
        self.device = device
        self.batch_size = batch_size
        
        # If shared_encoder is provided, we use its embed_dim if possible
        if shared_encoder is not None:
             # Most of our encoders (GAT, GGAC, etc.) have 'embed_dim' or nested attributes.
             try:
                 val = None
                 if hasattr(shared_encoder, 'embed_dim'):
                     val = shared_encoder.embed_dim
                 elif hasattr(shared_encoder, 'layers'):
                     val = shared_encoder.layers[0].att.module.embed_dim
                 
                 if isinstance(val, int):
                     hidden_dim = val
             except:
                 pass
             
        self.hidden_dim = hidden_dim
        self.input_dim_dynamic = input_dim_dynamic
        self.global_input_dim = global_input_dim
        self.critical_threshold = critical_threshold
        
        # 1. Temporal Encoder (Long Short-Term Memory)
        # Inputs: (Batch, N, History)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)
        
        # 2. Feature Fusion
        # Static (2) + Temporal Embedding (lstm_hidden) -> Hidden
        self.feature_embedding = nn.Linear(input_dim_static + lstm_hidden, hidden_dim)
        
        # 3. Spatial Encoder (GAT / Transformer)
        # Using TransformerEncoder as standard "GAT" on fully connected graph
        if shared_encoder is not None:
            self.gat_encoder = shared_encoder
        else:
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
        

        # Buffers for PPO
        self.clear_memory()

        # Force the Manager to prefer Routing (1) over Skipping (0) at the start
        with torch.no_grad():
            # Assuming gate_head's last layer is a Linear layer
            self.gate_head[-1].bias.fill_(0)
            
            # Revert to Neutral Bias
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
        Returns: (Batch, N, Hidden), (Batch, N, lstm_hidden)
        """
        B, N, H_len = dynamic.size()
        
        # LSTM Temporal Encoding
        # Reshape dynamic for LSTM: (Batch*N, History, 1)
        dyn_flat = dynamic.view(B * N, H_len, 1)
        _, (h_n, _) = self.lstm(dyn_flat)
        temporal_embed = h_n.squeeze(0).view(B, N, -1)
        
        # Concatenate with static
        # static: (Batch, N, 2)
        combined = torch.cat([static, temporal_embed], dim=2)
        
        # Project to hidden
        x = self.feature_embedding(combined)
        
        return x, temporal_embed

    def forward(self, static, dynamic, global_features):
        """
        Returns logits for mask and gate, and state value.
        global_features: (Batch, global_input_dim)
        """
        # Encode
        x, temporal_embed = self.feature_processing(static, dynamic) # (B, N, H)
        
        # Spatial Context
        # Transformer expects (Batch, Seq, F) with batch_first=True
        h = self.gat_encoder(x) # (B, N, H)
        
        # SKIP CONNECTION: Concatenate temporal_embed (LSTM output) to h
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

    def select_action(self, static, dynamic, global_features=None, deterministic=False, threshold=0.5, mask_threshold=0.5, target_mask=None):
        """
        static: (Batch, N, 2) - Locations
        dynamic: (Batch, N, 1) - Waste levels
        global_features: (Batch, 3) - Global context
        threshold: float - Probability threshold for gate=1 (Route) when deterministic
        mask_threshold: float - Probability threshold for mask=1 (Unmask) when deterministic
        target_mask: (Batch, N) of 0/1 or None - Expert target for mask auxiliary loss
        """
        mask_logits, gate_logits, value = self.forward(static, dynamic, global_features)
        
        # 1. Gate Decision (0=Skip, 1=Route)
        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_dist = torch.distributions.Categorical(gate_probs)
        
        if deterministic:
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

