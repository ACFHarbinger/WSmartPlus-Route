import math
import torch
import random
import numpy as np

from collections import defaultdict
from backend.src.utils.graph_utils import find_longest_path


class WeightContextualBandit:
    """
    A contextual bandit approach for dynamically selecting weight configurations
    in a waste collection reinforcement learning problem.
    """
    def __init__(self, 
                 num_days=10,
                 distance_matrix=None,
                 initial_weights=None,
                 context_features=None,
                 features_aggregation='avg',
                 exploration_strategy='ucb',
                 exploration_factor=0.5,
                 num_weight_configs=10,
                 weight_ranges=None,
                 window_size=20):
        """
        Initialize the Contextual Bandit for weight selection.
        
        Args:
            initial_weights (dict): Initial weight configuration {component: value}
            context_features (list): List of context feature names to use
            exploration_strategy (str): 'ucb', 'thompson_sampling', or 'epsilon_greedy'
            exploration_factor (float): Controls exploration vs. exploitation balance
            num_weight_configs (int): Number of weight configurations to generate
            weight_ranges (dict): Range for each weight component {component: (min, max)}
            window_size (int): Size of window for moving average reward calculation
        """
        self.num_days = num_days
        self.dist_matrix = distance_matrix
        
        # Generate initial weight configurations
        self.weight_ranges = weight_ranges
        self.weight_configs = self._generate_weight_configs(initial_weights, num_weight_configs)
        self.num_configs = len(self.weight_configs)
        features = []
        for context_feature in context_features:
            if context_feature in initial_weights.keys():
                features.append(f"{features_aggregation}_{context_feature}")
            else:
                features.append(context_feature)
        self.context_features = features
        self.features_aggregation = features_aggregation

        # Set exploration strategy
        self.exploration_strategy = exploration_strategy
        self.exploration_factor = exploration_factor

        # For Thompson sampling
        self.alpha = np.ones(self.num_configs)
        self.beta = np.ones(self.num_configs)
        
        # For UCB
        self.trials = np.zeros(self.num_configs)
        self.total_trials = 0
        
        # For reward tracking
        self.rewards = defaultdict(list)
        self.window_size = window_size
        
        # Current configuration
        self.max_feat_values = {}
        self.current_config_idx = 0
        self.current_config = self.weight_configs[0]
        
        # For context clustering
        self.contexts = []
        self.context_rewards = defaultdict(lambda: defaultdict(list))
        
        # Logging
        self.history = []
    
    def _generate_weight_configs(self, initial_weights, num_configs):
        """
        Generate a set of weight configurations to explore.
        
        Args:
            initial_weights (dict): Initial weight configuration
            num_configs (int): Number of configurations to generate
            
        Returns:
            list: List of weight configurations
        """
        configs = []
        
        # Add the initial configuration if provided
        if initial_weights:
            configs.append(initial_weights.copy())
        
        # Generate diverse configurations
        components = list(self.weight_ranges.keys())
        
        # Strategy 1: Vary one component at a time
        for component in components:
            min_val, max_val = self.weight_ranges[component]
            # Create variations of the initial weights
            base_config = initial_weights.copy() if initial_weights else {c: 1.0 for c in components}
            
            # Low value for this component
            low_config = base_config.copy()
            low_config[component] = min_val + 0.1 * (max_val - min_val)
            configs.append(low_config)
            
            # High value for this component
            high_config = base_config.copy()
            high_config[component] = max_val - 0.1 * (max_val - min_val)
            configs.append(high_config)
        
        # Strategy 2: Random combinations
        while len(configs) < num_configs:
            config = {component: random.uniform(self.weight_ranges[component][0], 
                                                self.weight_ranges[component][1]) 
                     for component in components}
            configs.append(config)
        
        return configs[:num_configs]
    
    def set_max_feature_values(self, mf_dict={}):
        tmp_mf_dict =  {}
        mf_keys = mf_dict.keys()
        for feature in self.context_features:
            n_vertices = self.dist_matrix.size(0)
            if feature not in mf_keys:
                if feature in ['overflow', 'waste']:
                    tmp_mf_dict[feature] = 1.0 * n_vertices
                elif feature == 'visited_ratio':
                    tmp_mf_dict[feature] = 1.0
                elif feature == 'length':
                    longest_length, _ = find_longest_path(self.dist_matrix)
                    tmp_mf_dict[feature] == longest_length
                else:
                    assert feature == 'day'
                    tmp_mf_dict[feature] = self.num_days
            else:
                tmp_mf_dict[feature] = mf_dict[feature]
        self.max_feat_values = tmp_mf_dict
    
    def get_current_weights(self):
        """
        Get the current weight configuration.
        
        Returns:
            dict: Current weight configuration
        """
        return self.current_config
    
    def _get_context_features(self, dataset, init_value=0.5):
        """
        Extract context features from the dataset.
        
        Args:
            dataset: Dataset containing problem instances
            
        Returns:
            dict: Dictionary of context features
        """
        def _set_context_value(feat_name, feature, context, add_agg=True):
            if isinstance(feature, torch.Tensor):
                if self.features_aggregation == 'avg':
                    feat_value = feature.mean().item()
                elif self.features_aggregation == 'sum':
                    feat_value = feature.sum().item()
                else:
                    assert self.features_aggregation == 'max'
                    feat_value = feature.max().item()
            else:
                feat_value = feature
            context_name = f"{self.features_aggregation}_{feat_name}" if add_agg else feat_name
            return {**context, context_name: feat_value}

        context = {}
        for fkey in self.context_features:
            if fkey in ['waste', 'overflow']:
                waste_levels = torch.stack([instance['waste'] for instance in dataset.data])
            
            if fkey == 'waste':
                context = _set_context_value("waste", waste_levels, context)
            elif fkey == 'overflow':
                max_waste = torch.stack([instance['max_waste'] for instance in dataset.data])
                overflow_mask = waste_levels >= max_waste
                context = _set_context_value("overflow", overflow_mask.float(), context)
            elif fkey == 'length' and not 'length' in context:
                context = _set_context_value("length", init_value, context) # Normalized value
            elif fkey == 'visited_ratio' and not 'visited_ratio' in context:
                context = _set_context_value("visited_ratio", init_value, context, add_agg=False) # Normalized value
            else:
                assert fkey == 'day'
                context = _set_context_value("day", len(self.history), context, add_agg=False)
        return context
    
    def _context_to_key(self, context):
        """
        Convert context to a discrete key for context clustering.
        
        Args:
            context (dict): Context features
            
        Returns:
            tuple: Discretized context key
        """
        discretized = []
        n_vertices = self.dist_matrix.size(0)
        for feature in self.context_features:
            if feature in context:
                value = context[feature]
                max_value = self.max_feat_values[feature]
                if feature in ['waste', 'overflow', 'length', 'visited_ratio', 'day']:
                    normalized = min(value / max_value, 1.0)
                else:
                    normalized = value
                
                # Discretize to vertices
                bin_idx = min(int(normalized * n_vertices), n_vertices - 1)
                discretized.append(bin_idx)
            else:
                discretized.append(0)
        return tuple(discretized)
    
    def get_current_weights(self, dataset):
        """
        Select a weight configuration based on the current context.
        
        Args:
            dataset: Dataset containing problem instances
            
        Returns:
            dict: Selected weight configuration
        """
        # Extract context features
        context = self._get_context_features(dataset)
        context_key = self._context_to_key(context)
        
        # Store context for learning
        self.contexts.append(context)
        
        # Select configuration based on exploration strategy
        if self.exploration_strategy == 'ucb':
            selected_idx = self._select_ucb(context_key)
        elif self.exploration_strategy == 'thompson_sampling':
            selected_idx = self._select_thompson_sampling(context_key)
        elif self.exploration_strategy == 'epsilon_greedy':
            selected_idx = self._select_epsilon_greedy(context_key)
        else:
            # Default to UCB
            selected_idx = self._select_ucb(context_key)
        
        # Update current configuration
        self.current_config_idx = selected_idx
        self.current_config = self.weight_configs[selected_idx]
        
        # Log selection
        self.history.append({
            'day': context.get('day', len(self.history)),
            'context': context,
            'selected_config': self.current_config,
            'selected_idx': selected_idx
        })
        return self.current_config
    
    def _select_ucb(self, context_key):
        """
        Select configuration using Upper Confidence Bound (UCB) strategy.
        
        Args:
            context_key: Discretized context key
            
        Returns:
            int: Index of selected configuration
        """
        # If we have context-specific data, use it
        if context_key in self.context_rewards and sum(self.trials) > 0:
            # Calculate UCB scores for each configuration
            ucb_scores = np.zeros(self.num_configs)
            
            for i in range(self.num_configs):
                rewards = self.context_rewards[context_key][i]
                
                if len(rewards) > 0:
                    # Mean reward
                    mean_reward = np.mean(rewards)
                    # Confidence term
                    confidence = self.exploration_factor * math.sqrt(math.log(self.total_trials) / max(1, self.trials[i]))
                    ucb_scores[i] = mean_reward + confidence
                else:
                    # If no data for this config, use a high value to encourage exploration
                    ucb_scores[i] = 1e6
            
            # Select the configuration with the highest UCB score
            return np.argmax(ucb_scores)
        else:
            # If no context data, explore uniformly
            return random.randint(0, self.num_configs - 1)
    
    def _select_thompson_sampling(self, context_key):
        """
        Select configuration using Thompson Sampling strategy.
        
        Args:
            context_key: Discretized context key
            
        Returns:
            int: Index of selected configuration
        """
        # If we have context-specific data, use it
        if context_key in self.context_rewards and any(len(self.context_rewards[context_key][i]) > 0 for i in range(self.num_configs)):
            # Sample from Beta distributions for each configuration
            samples = np.zeros(self.num_configs)
            for i in range(self.num_configs):
                rewards = self.context_rewards[context_key][i]
                if len(rewards) > 0:
                    # Normalize rewards to [0, 1] for Beta distribution
                    normalized_rewards = [(r - min(rewards)) / (max(rewards) - min(rewards) + 1e-6) for r in rewards]
                    successes = sum(normalized_rewards)
                    failures = len(normalized_rewards) - successes
                    
                    # Sample from Beta distribution
                    alpha = 1 + successes
                    beta = 1 + failures
                    samples[i] = np.random.beta(alpha, beta)
                else:
                    # If no data, use a high prior to encourage exploration
                    samples[i] = np.random.beta(1, 1)
            
            # Select the configuration with the highest sample
            return np.argmax(samples)
        else:
            # If no context data, explore uniformly
            return random.randint(0, self.num_configs - 1)
    
    def _select_epsilon_greedy(self, context_key):
        """
        Select configuration using Epsilon-Greedy strategy.
        
        Args:
            context_key: Discretized context key
            
        Returns:
            int: Index of selected configuration
        """
        # With probability epsilon, explore randomly
        if random.random() < self.exploration_factor:
            return random.randint(0, self.num_configs - 1)
        
        # Otherwise, exploit the best known configuration
        if context_key in self.context_rewards and any(len(self.context_rewards[context_key][i]) > 0 for i in range(self.num_configs)):
            mean_rewards = np.zeros(self.num_configs)
            
            for i in range(self.num_configs):
                rewards = self.context_rewards[context_key][i]
                
                if len(rewards) > 0:
                    mean_rewards[i] = np.mean(rewards)
                else:
                    mean_rewards[i] = -np.inf
            
            # Select the configuration with the highest mean reward
            return np.argmax(mean_rewards)
        else:
            # If no context data, explore uniformly
            return random.randint(0, self.num_configs - 1)
    
    def update(self, reward, cost_components, context=None, epsilon_params=(0.01, 1.)):
        """
        Update the bandit with the observed reward.
        
        Args:
            reward (float): Observed reward
            cost_components (dict): Dictionary of cost components
            context (dict, optional): Context features (if not provided, use the last context)
            
        Returns:
            dict: Current statistics
        """
        # Get context key
        if context is None and self.contexts: context = self.contexts[-1]
        context_key = self._context_to_key(context) if context else None
        
        # Update rewards
        if context_key:
            self.context_rewards[context_key][self.current_config_idx].append(reward)
            
            # Keep only the last window_size rewards
            if len(self.context_rewards[context_key][self.current_config_idx]) > self.window_size:
                self.context_rewards[context_key][self.current_config_idx] = self.context_rewards[context_key][self.current_config_idx][-self.window_size:]
        
        # Update trials
        self.trials[self.current_config_idx] += 1
        self.total_trials += 1
        
        # Update parameters for Thompson sampling
        # Normalize reward to [0, 1] for Beta distribution
        normalized_reward = max(0, min(1, (reward + 10) / 20))  # Assuming reward range [-10, 10]
        self.alpha[self.current_config_idx] = self.alpha[self.current_config_idx] + normalized_reward
        self.beta[self.current_config_idx] = self.beta[self.current_config_idx] + (1 - normalized_reward)
        
        # Decay epsilon for epsilon-greedy
        if self.exploration_strategy == 'epsilon_greedy':
            self.exploration_factor = max(epsilon_params[0], self.exploration_factor * epsilon_params[1])
        
        # Return current statistics
        stats = {
            'current_config': self.current_config,
            'current_config_idx': self.current_config_idx,
            'trials': self.trials.tolist(),
            'total_trials': self.total_trials,
            'reward': reward
        } 
        return stats
    
    def get_best_config(self, context=None):
        """
        Get the best configuration based on observed rewards.
        
        Args:
            context (dict, optional): Context features
            
        Returns:
            dict: Best weight configuration
        """
        if context:
            context_key = self._context_to_key(context)
            if context_key in self.context_rewards:
                mean_rewards = np.zeros(self.num_configs)
                for i in range(self.num_configs):
                    rewards = self.context_rewards[context_key][i]
                    if len(rewards) > 0:
                        mean_rewards[i] = np.mean(rewards)
                    else:
                        mean_rewards[i] = -np.inf
                
                best_idx = np.argmax(mean_rewards)
                return self.weight_configs[best_idx]
        
        # If no context or no data for context, return overall best
        if sum(self.trials) > 0:
            mean_rewards = np.zeros(self.num_configs)
            for i in range(self.num_configs):
                rewards = []
                for context_key in self.context_rewards:
                    rewards.extend(self.context_rewards[context_key][i])
                
                if rewards:
                    mean_rewards[i] = np.mean(rewards)
                else:
                    mean_rewards[i] = -np.inf
            
            best_idx = np.argmax(mean_rewards)
            return self.weight_configs[best_idx]
        
        # If no data at all, return current config
        return self.current_config
