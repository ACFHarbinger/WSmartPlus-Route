"""
Meta-Learning Module for Adaptive Weight Optimization.

This module provides a collection of "Meta-Learner" strategies designed to dynamically adjust
the objective weights (e.g., Cost vs. Waste vs. Time) during the Reinforcement Learning training process.
By adapting these weights, the agent can learn robust policies that balance conflicting objectives
effectively across different stages of training or different problem environments.

Key Strategies:
- **Multi-Objective RL (`multi_objective.py`)**:
    - Class: `MORLWeightOptimizer`.
    - Implements Pareto-front exploration strategies to find a balance between competing objectives.
- **Neural Weight Optimization (`weight_optimizer.py`)**:
    - Class: `RewardWeightOptimizer`.
    - Uses a recurrent neural network (RNN/LSTM) or other meta-models to predict optimal weight
      adjustments based on training history.
- **Contextual Bandits (`contextual_bandits.py`)**:
    - Class: `WeightContextualBandit`.
    - treats weight selection as a bandit problem, using algorithms like UCB or Thompson Sampling
      to select weights given the current "context" (state of the environment).
- **Temporal Difference Learning (`temporal_difference_learning.py`)**:
    - Class: `CostWeightManager`.
    - Adjusts weights based on TD-error signals, aiming to align the reward signal with long-term value.

Common Interface:
    All strategies implement the protocol/interface defined in `weight_strategy.py`, ensuring
    they can be plugged into the main training pipeline interchangeably.
    - `propose_weights(context)`: Suggest new weights for the next episode/batch.
    - `feedback(reward, metrics, ...)`: Receive feedback to update internal state.

Example Usage:
    >>> from logic.src.pipeline.reinforcement_learning.meta import MORLWeightOptimizer
    >>> optimizer = MORLWeightOptimizer(
    ...     initial_weights={'w_waste': 1.0, 'w_over': 2.0},
    ...     exploration_factor=0.2
    ... )
    >>> current_weights = optimizer.propose_weights(current_state_context)
"""

from .multi_objective import MORLWeightOptimizer
from .weight_optimizer import RewardWeightOptimizer
from .contextual_bandits import WeightContextualBandit
from .temporal_difference_learning import CostWeightManager