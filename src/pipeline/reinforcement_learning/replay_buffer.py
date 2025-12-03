import torch


class PPOReplayBuffer:
    def __init__(self, capacity, observation_shape, action_shape, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.observations = torch.zeros((capacity, *observation_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,), dtype=torch.bool, device=device)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.idx = 0
        self.size = 0
        
    def add(self, obs, action, log_prob, reward, value, done):
        # Store transition in buffer
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.log_probs[self.idx] = log_prob
        self.rewards[self.idx] = reward
        self.values[self.idx] = value
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def compute_advantages(self, last_value, gamma, lam):
        """Compute GAE advantages and returns for all stored transitions"""
        last_gae_lam = 0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[(t + 1) % self.capacity]
            next_non_terminal = 0.0 if t == self.size - 1 else 1.0 - float(self.dones[(t + 1) % self.capacity])
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            self.returns[t] = self.advantages[t] + self.values[t]
            
    def sample(self, batch_size):
        """Sample a mini-batch of transitions"""
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        return (
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.advantages[indices],
            self.returns[indices],
            self.values[indices]
        )
        
    def clear(self):
        self.idx = 0
        self.size = 0

    def train_with_replay_buffer(model, optimizer, baseline, env, replay_buffer, opts):
        # PPO hyperparameters
        gamma = opts.get('gamma', 0.99)
        lam = opts.get('lambda', 0.95)
        clip_eps = opts.get('clip_eps', 0.2)
        value_coef = opts.get('value_coef', 0.5)
        entropy_coef = opts.get('entropy_coef', 0.01)
        ppo_epochs = opts.get('ppo_epochs', 4)
        mini_batch_size = opts.get('mini_batch_size', 64)
        buffer_update_freq = opts.get('buffer_update_freq', 2048)  # Steps before updating buffer
        replay_ratio = opts.get('replay_ratio', 4)  # How many times to sample from buffer per new data
        
        obs = env.reset()
        episode_rewards = []
        step = 0
        
        while step < opts['total_steps']:
            # Collect new experiences
            for _ in range(buffer_update_freq):
                # Get action from policy
                with torch.no_grad():
                    action, log_prob, _ = model.act(obs)
                    value = baseline.predict(obs)
                
                # Execute action in environment
                next_obs, reward, done, _ = env.step(action)
                
                # Store transition in buffer
                replay_buffer.add(obs, action, log_prob, reward, value, done)
                
                # Update counters and variables
                step += 1
                episode_rewards.append(reward)
                
                # Reset if episode ended
                if done:
                    obs = env.reset()
                    # Log episode statistics
                    if len(episode_rewards) > 0:
                        print(f"Episode reward: {sum(episode_rewards)}")
                        episode_rewards = []
                else:
                    obs = next_obs
            
            # Compute advantages for the entire buffer
            with torch.no_grad():
                last_value = baseline.predict(obs)
            replay_buffer.compute_advantages(last_value, gamma, lam)
            
            # PPO update phase - train multiple epochs on replay buffer
            for _ in range(replay_ratio):
                # Normalize advantages in buffer
                buffer_advantages = replay_buffer.advantages[:replay_buffer.size]
                buffer_advantages = (buffer_advantages - buffer_advantages.mean()) / (buffer_advantages.std() + 1e-8)
                
                for _ in range(ppo_epochs):
                    # Sample mini-batch from buffer
                    batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_old_values = replay_buffer.sample(mini_batch_size)
                    
                    # Get current policy and value predictions
                    new_log_probs, entropy = model.evaluate_actions(batch_obs, batch_actions)
                    new_values = baseline.forward(batch_obs)
                    
                    # PPO policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value function loss
                    value_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -clip_eps, clip_eps)
                    vf_loss1 = (new_values - batch_returns).pow(2)
                    vf_loss2 = (value_clipped - batch_returns).pow(2)
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                    
                    entropy_loss = -entropy_coef * entropy.mean()
                    loss = policy_loss + value_coef * value_loss + entropy_loss
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['max_grad_norm'])
                    optimizer.step()

    observation_shape = (2,)  # Example shape
    action_shape = (1,)       # Example shape
    replay_buffer = PPOReplayBuffer(
        capacity=opts.get('buffer_capacity', 10000),
        observation_shape=observation_shape,
        action_shape=action_shape,
        device=opts['device']
    )

    for state, action, log_prob, reward, next_state, done in collected_experiences:
        replay_buffer.add(state, action, log_prob, reward, baseline.predict(state), done)

    # Add importance correction term to account for off-policy data
    importance_weight = current_log_prob - old_log_prob_from_buffer
    importance_weight = torch.clamp(torch.exp(importance_weight), 0.0, 10.0)  # Clip for stability
    corrected_advantage = importance_weight * advantage
    
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def sample_prioritized(self, batch_size, alpha=0.6, beta=0.4):
        """Sample with prioritization, where alpha controls the degree of prioritization"""
        if self.size == 0:
            return None
            
        # Compute sampling probabilities
        probs = self.priorities[:self.size] ** alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.advantages[indices],
            self.returns[indices],
            self.values[indices],
            weights,
            indices
        )