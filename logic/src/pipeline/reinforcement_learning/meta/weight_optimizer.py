"""
Neural Network-Based Meta-Learning for Reward Weight Optimization.

This module implements a deep learning approach to meta-learning where an RNN model
learns to predict optimal weight adjustments based on historical performance data.
The approach treats weight optimization as a sequence-to-sequence learning problem.

The neural meta-learner observes:
    - Historical weight configurations
    - Performance metrics (waste, overflows, distance)
    - Achieved rewards

And learns to predict:
    - Weight adjustments that improve future performance
    - Exploration strategies for new weight configurations
    - Adaptation patterns for different problem characteristics

Architecture:
    - Input: Concatenated [weights, performance_metrics] sequences
    - RNN Core: Captures temporal dependencies in weight-performance relationships
    - Output: Weight adjustment vectors
    - Loss: Weighted combination of reward prediction and exploration variance

Classes:
    RewardWeightOptimizer: RNN-based meta-learner for weight optimization

Key Features:
    - Automatic feature extraction from performance history
    - Batch-based meta-learning updates
    - Gradient clipping for training stability
    - Bounded weight constraints
    - Multiple optimizer support (Adam, AdamW, RAdam, etc.)

Example:
    optimizer = RewardWeightOptimizer(
        model_class=MetaRNN,
        initial_weights={'w_waste': 1.0, 'w_over': 2.0},
        history_length=10,
        hidden_size=64,
        lr=0.001
    )

    # During training
    weights = optimizer.propose_weights()
    # ... run episode ...
    optimizer.feedback(reward, metrics)
"""

import torch

from logic.src.pipeline.reinforcement_learning.meta.weight_strategy import (
    WeightAdjustmentStrategy,
)


class RewardWeightOptimizer(WeightAdjustmentStrategy):
    """
    Manager for the meta-learning process of optimizing reward weights
    based on historical performance data. Uses an RNN to propose updates.
    """

    def __init__(
        self,
        model_class,
        initial_weights,
        history_length=10,
        hidden_size=64,
        lr=0.001,
        device="cuda",
        meta_batch_size=8,
        min_weights=None,
        max_weights=None,
        meta_optimizer=None,
    ):
        """
        Args:
            initial_weights: Dict of weight names and initial values
            history_length: Number of time steps to consider for meta-learning
            hidden_size: Hidden size of the RNN
            lr: Learning rate for the meta-learner
            device: Device to run computations on
            meta_batch_size: Batch size for meta-learning updates
            min_weights: Minimum allowed values for weights
            max_weights: Maximum allowed values for weights
        """
        self.device = device
        self.weight_names = list(initial_weights.keys())
        self.num_weights = len(self.weight_names)
        self.history_length = history_length
        self.meta_batch_size = meta_batch_size

        # Initialize current weights
        self.current_weights = torch.tensor(list(initial_weights.values()), dtype=torch.float32, device=device)

        # Set weight constraints
        self.min_weights = torch.tensor(
            min_weights if min_weights else [0.0] * self.num_weights,
            dtype=torch.float32,
            device=device,
        )
        self.max_weights = torch.tensor(
            max_weights if max_weights else [float("inf")] * self.num_weights,
            dtype=torch.float32,
            device=device,
        )

        # Initialize feature extraction parameters
        self.num_perf_metrics = len(self.weight_names) + 1  # overflows, length, waste, total_reward

        # Define input size: current weights + performance metrics
        input_size = self.num_weights + self.num_perf_metrics

        # Initialize DNN meta-learner
        self.meta_model = model_class(input_size=input_size, hidden_size=hidden_size, output_size=self.num_weights).to(
            device
        )

        # Initialize optimizer for the meta-learner
        optimizer_params = [{"params": self.meta_model.parameters(), "lr": lr}]
        self.optimizer = {
            "adam": torch.optim.Adam(optimizer_params),
            "adamax": torch.optim.Adamax(optimizer_params),
            "adamw": torch.optim.AdamW(optimizer_params),
            "radam": torch.optim.RAdam(optimizer_params),
            "nadam": torch.optim.NAdam(optimizer_params),
            "rmsprop": torch.optim.RMSprop(optimizer_params),
        }.get(meta_optimizer, None)
        assert self.optimizer is not None, "Unknown optimizer: {}".format(meta_optimizer)

        # Initialize histories for weights and performance
        self.weight_history = []
        self.performance_history = []
        self.reward_history = []

        # Keep track of step for logging
        self.meta_step = 0

    def propose_weights(self, context=None):
        """
        Implementation of Strategy interface.
        Returns the current weights, potentially updated if history is sufficient.
        """
        # Internal update logic that was previously in main loop or update_weights
        self.update_weights_internal()
        return self.get_current_weights()

    def feedback(self, reward, metrics, day=None, step=None):
        """
        Implementation of Strategy interface.
        Updates history and performs a meta-learning step.
        """
        if isinstance(metrics, dict):
            # Fallback logic to convert dict to list matching weight order, similar to original code assumption
            # Original code in loop: list(c_dict.values()) + list(l_dict.values())
            # Here we trust the caller passes a tensor or list if possible, or we might need to be specific.
            # Ideally the Adapter/Trainer handles the conversion.
            # If metrics is passed as a dict, we blindly convert values to list, assuming compatibility.
            # This is risky but maintains previous behavior if caller is refactored carefully.
            perf_values = list(metrics.values())
        elif isinstance(metrics, (list, tuple)):
            perf_values = metrics
        elif isinstance(metrics, torch.Tensor):
            perf_values = metrics
        else:
            raise ValueError("metrics must be a list or Tensor")

        self.update_histories(perf_values, reward)
        self.meta_learning_step()

    def update_histories(self, performance_metrics, reward):
        """
        Update histories with current weights and performance metrics

        Args:
            performance_metrics: List of the cost function terms
            reward: Total reward achieved with current weights
        """
        self.weight_history.append(self.current_weights.clone().cpu())
        # Ensure tensor
        p_tensor = torch.as_tensor(performance_metrics, dtype=torch.float32)
        self.performance_history.append(p_tensor)
        self.reward_history.append(reward)

        # Keep history at desired length
        if len(self.weight_history) > self.history_length:
            self.weight_history.pop(0)
            self.performance_history.pop(0)
            self.reward_history.pop(0)

    def prepare_meta_learning_batch(self):
        """
        Prepare batch for meta-learning by combining weights and performance history

        Returns:
            features: Tensor of shape [batch_size, seq_len, input_size]
            targets: Tensor of target rewards
        """
        if len(self.weight_history) < 2:
            return None, 0

        # Calculate sequence length based on available history
        seq_len = min(self.history_length, len(self.weight_history) - 1)

        # Create features by combining weights and performance metrics
        features = []
        targets = []

        # Start from different points to create a batch
        for start_idx in range(len(self.weight_history) - seq_len):
            # Create sequence of [weights, performance_metrics]
            seq_features = []
            for i in range(start_idx, start_idx + seq_len):
                # Combine weights and performance metrics
                # Ensure all on same device
                w_hist = self.weight_history[i].to(self.device)
                p_hist = self.performance_history[i].to(self.device)

                # Check shapes: performance metrics might be (N,) or (1, N) or (Batch, N) if sloppy
                if p_hist.dim() > 1:
                    p_hist = p_hist.view(-1)

                combined = torch.cat([w_hist, p_hist])
                seq_features.append(combined)

            # Target is the reward after sequence
            target = self.reward_history[start_idx + seq_len]

            features.append(torch.stack(seq_features))
            targets.append(target)

            # Break if we have enough samples
            if len(features) >= self.meta_batch_size:
                break

        # Stack into tensors
        if features:
            features = torch.stack(features).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
            return features, targets

        return None, 0

    def meta_learning_step(self):
        """
        Perform a meta-learning step to update the RNN model

        Returns:
            loss: Training loss for this step, or None if not enough data
        """
        features, targets = self.prepare_meta_learning_batch()

        if features is None:
            return None

        # Set model to training mode
        self.meta_model.train()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        weight_adjustments, _ = self.meta_model(features)

        # Predict next weights
        pred_weights = features[:, -1, : self.num_weights] + weight_adjustments
        pred_weights = torch.clamp(pred_weights, self.min_weights, self.max_weights)

        # Use MSE loss between predicted weights and ideal weights
        # Here we're assuming that weights leading to higher rewards are better
        # So we scale the loss by the negative reward
        weights_variance = torch.var(pred_weights, dim=0).sum()
        reward_pred = -targets  # Negative because we want to maximize reward

        # Weighted loss: optimize for reward while encouraging exploration
        alpha = 0.8  # Balance between reward optimization and exploration
        loss = alpha * reward_pred.mean() + (1 - alpha) * weights_variance

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Log metrics
        self.meta_step += 1

        return loss.item()

    def recommend_weights(self):
        """
        Generate weight adjustment recommendations based on history

        Returns:
            New weights after applying recommended adjustments
        """
        if len(self.weight_history) < self.history_length:
            # Not enough history, return current weights
            return self.current_weights

        # Set model to evaluation mode
        self.meta_model.eval()

        with torch.no_grad():
            # Prepare input sequence from recent history
            seq_features = []
            for i in range(-self.history_length, 0):
                # Combine weights and performance metrics
                w_hist = self.weight_history[i].to(self.device)
                p_hist = self.performance_history[i].to(self.device)
                if p_hist.dim() > 1:
                    p_hist = p_hist.view(-1)

                combined = torch.cat([w_hist, p_hist])
                seq_features.append(combined)

            # Create batch with single sequence
            features = torch.stack(seq_features).unsqueeze(0).to(self.device)

            # Get weight adjustment recommendations
            weight_adjustments, _ = self.meta_model(features)

            # Apply adjustments to current weights
            new_weights = self.current_weights + weight_adjustments.squeeze(0)

            # Ensure weights stay within bounds
            new_weights = torch.clamp(new_weights, self.min_weights, self.max_weights)

            return new_weights

    def update_weights_internal(self, force_update=False):
        """
        Update current weights based on RNN recommendations

        Args:
            force_update: Whether to force an update even with limited history

        Returns:
            Boolean indicating if weights were updated
        """
        if len(self.weight_history) < self.history_length and not force_update:
            return False

        # Get recommended weights
        new_weights = self.recommend_weights()

        # Update current weights
        self.current_weights = new_weights
        return True

    def get_current_weights(self):
        """
        Get current weight values as a dictionary

        Returns:
            Dictionary mapping weight names to values
        """
        return {name: self.current_weights[i].item() for i, name in enumerate(self.weight_names)}
