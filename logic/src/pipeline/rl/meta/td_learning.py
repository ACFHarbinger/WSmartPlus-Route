"""
Cost Weight Manager using Temporal Difference (TD) Learning.
Robust implementation for adaptive multi-objective weight management.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy


class CostWeightManager(WeightAdjustmentStrategy):
    """
    Adjusts cost weights (w_lost, w_waste, w_length, w_overflows) using
    tabular Temporal Difference (TD) Learning.

    The state is defined by the current weight configuration (discretized).
    The action is a perturbation to one or more weights.
    The reward is the improvement in validation efficiency (kg/km) or negative loss.

    Attributes:
        learning_rate (float): Step size for value updates (alpha).
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        n_bins (int): Number of bins for discretizing weights.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        n_bins: int = 20,
        weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs,
    ):
        """Initialize TDLearningWeightOptimizer."""
        super().__init__()
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_bins = n_bins

        # Default weight bounds if not provided
        self.weight_bounds = weight_bounds or {
            "w_lost": (0.0, 10.0),
            "w_waste": (0.0, 10.0),
            "w_length": (0.0, 1.0),
            "w_overflows": (0.0, 20.0),
        }

        # Current weights (initialized to mid-points or arguably reasonable defaults)
        self.weights = {"w_lost": 1.0, "w_waste": 1.0, "w_length": 0.05, "w_overflows": 5.0}

        # Value table V(s) mapping discretized_state -> value
        self.values: Dict[Tuple[int, ...], float] = {}

        # Trace for eligibility traces or simple 1-step
        self.last_state: Optional[Tuple[int, ...]] = None
        self.last_val: float = 0.0

        # History for logging
        self.history = []

    def get_current_weights(self) -> Dict[str, float]:
        """Return the current continuous weights."""
        return self.weights.copy()

    def propose_weights(self, context=None) -> Dict[str, float]:
        """
        Propose weights for the next epoch.
        This is where the 'Action' is taken (changing weights).
        """
        # Current state
        self._discretize(self.weights)

        # Exploration: Random perturbation
        if random.random() < self.epsilon:
            action_key = random.choice(list(self.weights.keys()))
            delta = random.choice([-0.1, 0.1]) * (self.weight_bounds[action_key][1] - self.weight_bounds[action_key][0])
            self._apply_change(action_key, delta)
        else:
            # Exploitation: Greedy hill-climbing on V(s)?
            # In tabular TD for parameter tuning, usually we assume the current state IS the param config.
            # So "action" is moving to a neighbor state.
            # We look at neighbors and pick the one with highest V(s').
            best_neighbor_weights = self.weights.copy()
            best_val = -float("inf")

            # Simple local search (coordinate descent style)
            found_better = False
            for k in self.weights:
                step = (self.weight_bounds[k][1] - self.weight_bounds[k][0]) * 0.05
                for sign in [-1, 1]:
                    test_weights = self.weights.copy()
                    val = test_weights[k] + sign * step
                    # Clip
                    val = max(self.weight_bounds[k][0], min(self.weight_bounds[k][1], val))
                    test_weights[k] = val

                    state_key = self._discretize(test_weights)
                    v_est = self.values.get(state_key, 0.0)  # Optimistic init?

                    if v_est > best_val:
                        best_val = v_est
                        best_neighbor_weights = test_weights
                        found_better = True

            if found_better:
                self.weights = best_neighbor_weights

        # Store state for update in feedback() step
        self.last_state = self._discretize(self.weights)
        return self.weights

    def feedback(self, reward: float, metrics: List = None, day: int = None):
        """
        Update V(s) based on the observed reward signal.

        V(s_t) <- V(s_t) + alpha * [R_{t+1} + gamma * V(s_{t+1}) - V(s_t)]

        Note: In this architecture, 'feedback' is called after training with 'propose_weights' settings.
        So 'last_state' is s_t. 'reward' is R_{t+1}.
        The 'next state' is effectively the same unless we changed it again?
        Actually, we update V(last_state) based on the reward we just got.
        """
        if self.last_state is None:
            return

        # Get current state (which was the state used to generate this reward)
        # Wait, usually: Action (Weights) -> Reward.
        # So V(s) measures how good weight config 's' is.
        # This looks more like Bandit or direct Value Learning than typical RL transition.
        # If we treat it as calculating the Value of a parameter setting:
        # V(s) = (1-lr)*V(s) + lr*Reward
        # This is essentially a moving average of rewards for state s.
        # If we assume transition dynamics (s -> s'), then we use gamma.
        # Here, the transition is determined by our 'propose_weights' logic.

        # Simple TD(0) update
        current_val = self.values.get(self.last_state, 0.0)

        # Since we don't know s_{t+1} yet (we haven't chosen next weights),
        # we can't do full SARSA/Q-Learning update here easily unless we defer.
        # BUT, standard approach for hyperparam tuning is to treat Value as "Expected Reward of Data".
        # So we just update the value estimate of the visited node.

        # Update: V(s) = V(s) + alpha * (Reward - V(s))
        # (This is Monte Carlo update if gamma=0, or averaging).
        # Let's assume weights persist, so we are in a continuous episode.

        new_val = current_val + self.lr * (reward - current_val)
        self.values[self.last_state] = new_val

        self.history.append(
            {"state": self.last_state, "reward": reward, "value": new_val, "weights": self.weights.copy()}
        )

        # Decrease epsilon over time
        self.epsilon *= 0.995

    def _discretize(self, weights: Dict[str, float]) -> Tuple[int, ...]:
        """
        Convert continuous weights to a discrete tuple key.
        Sorts keys to ensure consistent ordering.
        """
        keys = sorted(weights.keys())
        indices = []
        for k in keys:
            val = weights[k]
            low, high = self.weight_bounds[k]
            # Normalize to 0-1
            norm = (val - low) / (high - low + 1e-8)
            norm = max(0.0, min(1.0, norm))
            idx = int(norm * self.n_bins)
            indices.append(idx)
        return tuple(indices)

    def _apply_change(self, key: str, delta: float):
        """Apply additive change to weight with clipping."""
        low, high = self.weight_bounds[key]
        self.weights[key] = max(low, min(high, self.weights[key] + delta))

    def state_dict(self):
        """Get state dictionary."""
        return {"values": self.values, "weights": self.weights, "epsilon": self.epsilon}

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.values = state_dict["values"]
        self.weights = state_dict["weights"]
        self.epsilon = state_dict.get("epsilon", self.epsilon)
