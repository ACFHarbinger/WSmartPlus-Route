"""
Temporal Difference Learning for Cost Weight Adaptation.

This module implements TD-based meta-learning for dynamically adjusting cost weights
during reinforcement learning training. The approach uses temporal difference errors
to update weights based on their contribution to the observed reward signal.

The TD learning approach is particularly effective for:
    - Online weight adaptation during training
    - Balancing multiple cost components (waste, overflows, distance)
    - Handling non-stationary reward distributions
    - Learning from sparse feedback signals

Key Concepts:
    - TD Error: Difference between observed and expected reward
    - Weight Update: Proportional to TD error and component contribution
    - Learning Rate Decay: Gradual reduction of adaptation rate over time
    - Moving Average Baseline: Smoothed expected reward estimation

Classes:
    CostWeightManager: Main TD-based weight adjustment strategy

Functions:
    update_cost_weights_td: Core TD learning update rule for weight adjustment

Example:
    manager = CostWeightManager(
        initial_weights={'waste': 1.0, 'over': 2.0, 'len': 0.5},
        learning_rate=0.01,
        decay_rate=0.999
    )

    # After each step
    weights = manager.update_weights(observed_reward, cost_components)
"""

from logic.src.pipeline.reinforcement_learning.meta.weight_strategy import (
    WeightAdjustmentStrategy,
)


def update_cost_weights_td(
    cost_weights,
    cost_components,
    td_error,
    learning_rate=0.01,
    weight_ranges=[0.01, 5.0],
):
    """
    Update cost weights using Temporal Difference (TD) learning.

    Args:
        cost_weights (dict): Current weights for different cost components
        cost_components (dict): Observed cost components for the current step
        td_error (float): Temporal difference error (observed reward - expected reward)
        learning_rate (float): Learning rate for weight adjustment
        weight_ranges (list): Range for weight components [min, max]

    Returns:
        dict: Updated cost weights
    """
    # Create a copy of current weights to update
    new_weights = cost_weights.copy()

    # Update weights based on TD error and their contribution to the total cost
    for component, value in cost_components.items():
        if component in new_weights:
            # For components we want to maximize (e.g., waste collected),
            # increase weight when TD error is positive
            if component == "waste":
                weight_update = learning_rate * td_error * value
            # For components we want to minimize (e.g., overflows, length),
            # decrease weight when TD error is positive
            else:
                weight_update = -learning_rate * td_error * value

            # Update the weight
            new_weights[component] += weight_update

            # Ensure weights stay within bounds
            new_weights[component] = max(weight_ranges[0], min(new_weights[component], weight_ranges[1]))
    return new_weights


class CostWeightManager(WeightAdjustmentStrategy):
    """
    Manages the dynamic adjustment of cost weights based on TD learning.
    """

    def __init__(
        self,
        initial_weights,
        learning_rate=0.01,
        decay_rate=0.999,
        weight_ranges=[0.01, 5.0],
        window_size=10,
    ):
        """
        Initialize the CostWeightManager.

        Args:
            initial_weights (dict): Initial weights for cost components
            learning_rate (float): Initial learning rate for weight adjustments
            decay_rate (float): Rate at which learning rate decays over time
            weight_ranges (list): Range for weight components [min, max]
            window_size (int): Size of window for moving average reward calculation
        """
        self.weights = initial_weights.copy()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.weight_ranges = weight_ranges
        self.window_size = window_size

        # For tracking rewards and creating TD error
        self.day = 0
        self.past_rewards = []
        self.expected_reward = None

    def propose_weights(self, context=None):
        """
        Implementation of Strategy interface.
        """
        return self.get_current_weights()

    def feedback(self, reward, metrics, day=None, step=None):
        """
        Implementation of Strategy interface.
        Update weights based on observed reward and cost components.
        """
        cost_components = {}
        if isinstance(metrics, dict):
            cost_components = metrics

        self.update_weights(reward, cost_components)

    def update_weights(self, observed_reward, cost_components):
        """
        Update weights based on observed reward and cost components.

        Args:
            observed_reward (float): The reward observed for the current action
            cost_components (dict): Dictionary of cost components (waste, over, len)

        Returns:
            dict: Updated weights
        """
        # Add the observed reward to our history
        self.past_rewards.append(observed_reward)
        if len(self.past_rewards) > self.window_size:
            self.past_rewards.pop(0)

        # Calculate expected reward (moving average of past rewards)
        if self.expected_reward is None:
            self.expected_reward = observed_reward
        else:
            # If we have history, update the expected reward
            avg_reward = sum(self.past_rewards) / len(self.past_rewards)
            self.expected_reward = 0.9 * self.expected_reward + 0.1 * avg_reward

        # Calculate TD error
        td_error = observed_reward - self.expected_reward

        # Update weights based on TD error
        self.weights = update_cost_weights_td(
            self.weights,
            cost_components,
            td_error,
            self.learning_rate,
            self.weight_ranges,
        )

        # Decay learning rate
        self.learning_rate *= self.decay_rate
        self.day += 1

        return self.weights

    def get_current_weights(self):
        """Get current weights."""
        return self.weights

    def get_learning_rate(self):
        """Get current learning rate."""
        return self.learning_rate
