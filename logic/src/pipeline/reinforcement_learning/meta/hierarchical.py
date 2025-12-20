import torch
import torch.nn as nn
import torch.nn.functional as F


class HRLManager(nn.Module):
    """
    Manager agent for Hierarchical Reinforcement Learning.
    Uses PPO to optimize separate cost weights (sub-goals) for the Worker agent.
    """
    def __init__(self,
                 initial_weights,
                 history_length=10,
                 hidden_size=128,
                 lr=3e-4,
                 device='cuda',
                 min_weights=None,
                 max_weights=None,
                 ppo_epochs=4,
                 clip_eps=0.2,
                 gamma=0.99,
                 lam=0.95):
        super().__init__()
        self.device = device
        self.history_length = history_length
        self.weight_names = list(initial_weights.keys())
        self.num_weights = len(self.weight_names)
        self.ppo_epochs = ppo_epochs
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

        # Initialize current weights
        self.current_weights = torch.tensor(list(initial_weights.values()), dtype=torch.float32, device=device)
        
        # Constraints
        self.min_weights = torch.tensor(min_weights if min_weights else [0.0] * self.num_weights, 
                                      dtype=torch.float32, device=device)
        self.max_weights = torch.tensor(max_weights if max_weights else [5.0] * self.num_weights, 
                                      dtype=torch.float32, device=device)

        # Observation space:
        # History of [weights (N) + performance metrics (N+1)] flattened
        # Performance metrics: values for each weight term + total reward
        metrics_per_step = self.num_weights + 1 
        input_size = (self.num_weights + metrics_per_step) * history_length

        # Input normalization
        self.state_mean = torch.zeros(input_size, device=device)
        self.state_std = torch.ones(input_size, device=device)
        self.state_n = 0
        
        # Actor Network (Gaussian Policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.num_weights)
        ).to(device)
        
        # Initialize output bias to initial_weights
        # This ensures exploration starts around the user-provided (likely working) configuration
        # Inverse tanh is not needed as final layer is Linear
        self.actor_mean[-1].bias.data = self.current_weights.clone()
        # Initialize weights to small values to reduce initial variance from the mean
        self.actor_mean[-1].weight.data *= 0.01

        # Initialize log_std to -1.0 (std approx 0.37) for reduced initial variance
        self.actor_log_std = nn.Parameter(torch.ones(1, self.num_weights).to(device) * -1.0)

        # Critic Network (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Buffers
        self.weight_history = []
        self.performance_history = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def _update_running_stats(self, x):
        """Update running mean and std."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        if self.state_n == 0:
            self.state_mean = batch_mean
            self.state_std = torch.sqrt(batch_var + 1e-8)
        else:
            delta = batch_mean - self.state_mean
            tot_count = self.state_n + batch_count
            
            new_mean = self.state_mean + delta * batch_count / tot_count
            m_a = self.state_std ** 2 * self.state_n
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + delta ** 2 * self.state_n * batch_count / tot_count
            new_var = m_2 / tot_count
            new_std = torch.sqrt(new_var + 1e-8)
            
            self.state_mean = new_mean
            self.state_std = new_std
        self.state_n += batch_count

    def get_state(self):
        """Construct normalized state from history."""
        if len(self.weight_history) < self.history_length:
            # Pad with zeros if not enough history
            padding_len = self.history_length - len(self.weight_history)
            
            # Construct padding
            w_pad = [torch.zeros(self.num_weights, device=self.device) for _ in range(padding_len)]
            p_pad = [torch.zeros(self.num_weights + 1, device=self.device) for _ in range(padding_len)]
            
            w_hist = w_pad + [w.to(self.device) for w in self.weight_history]
            p_hist = p_pad + [p.to(self.device) for p in self.performance_history]
        else:
            w_hist = [w.to(self.device) for w in self.weight_history[-self.history_length:]]
            p_hist = [p.to(self.device) for p in self.performance_history[-self.history_length:]]
        
        # Flatten history
        # Each step i: [w_i, p_i]
        flat_state = []
        for w, p in zip(w_hist, p_hist):
            flat_state.append(w)
            flat_state.append(p)
        
        raw_state = torch.cat(flat_state).unsqueeze(0) # (1, input_size)
        
        # Update stats and normalize
        # Note: We update stats continuously during training
        with torch.no_grad():
            self._update_running_stats(raw_state)
            norm_state = (raw_state - self.state_mean) / (self.state_std + 1e-8)
            
        return norm_state.clamp(-5.0, 5.0) # Clip extreme values

    def select_action(self):
        """Select new weights based on current state."""
        state = self.get_state()
        
        with torch.no_grad():
            action_mean = self.actor_mean(state)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            
            # Sample action
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
            # Clip action to valid range (for execution)
            clipped_action = torch.clamp(action, self.min_weights, self.max_weights)
            
            log_prob = dist.log_prob(action).sum(-1)
            value = self.critic(state)
            
        self.states.append(state)
        self.actions.append(action) # Store raw action for training
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.current_weights = clipped_action.squeeze(0)
        return {name: self.current_weights[i].item() for i, name in enumerate(self.weight_names)}
    
    def store_transition(self, metrics, reward, done=False):
        """Store the result of the action."""
        # Update history buffers for state construction
        self.weight_history.append(self.current_weights.clone())
        self.performance_history.append(torch.tensor(metrics, dtype=torch.float32, device=self.device))
        
        # Keep history finite (though state construction handles slice, this prevents unbounded growth)
        if len(self.weight_history) > self.history_length * 2:
            self.weight_history.pop(0)
            self.performance_history.pop(0)

        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        """Perform PPO update using stored transitions."""
        if len(self.rewards) < 2:
            return None

        # Convert buffers to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.cat(self.values).squeeze(-1).detach()
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # We don't have next_state value for the very last step if it's not terminal
        # But HRL steps are usually "continuous" until end of training meta-epochs
        # We can assume next_value is 0 or simple bootstrapping if we had next state
        # For now, standard GAE with 0 terminal value for last step
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0 # Or bootstrap if we had next state
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Epochs
        for _ in range(self.ppo_epochs):
            # Recalculate policy distributions
            action_mean = self.actor_mean(states)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            new_log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            
            # Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss
            new_values = self.critic(states).squeeze(-1)
            value_loss = F.mse_loss(new_values, returns)
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            
        # Clear training buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        return loss.item()

    def get_current_weights(self):
        """Interface compatibility."""
        return {name: self.current_weights[i].item() for i, name in enumerate(self.weight_names)}
