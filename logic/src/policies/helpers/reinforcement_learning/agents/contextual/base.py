"""
Base class definition for Contextual Multi-Armed Bandit (CMAB) agents.
"""

from abc import abstractmethod
from collections import deque
from typing import Any, Deque, Optional

import numpy as np

from logic.src.policies.helpers.reinforcement_learning.agents.base import RLAgent


class ContextualBanditAgent(RLAgent):
    """
    Base class for Contextual Multi-Armed Bandit (CMAB) agents.

    Unlike traditional bandits, contextual bandits use external features (context)
    to inform action selection. This base class extends the standard RLAgent
    with bookkeeping for CMAB-specific metrics and histories.
    """

    def __init__(self, n_arms: int, feature_dim: int, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the contextual bandit agent.

        Args:
            n_arms: Number of available actions.
            feature_dim: Dimension of the context feature vector.
            seed: Optional seed for the random number generator.
            history_size: Size of the reward tracking buffer.
        """
        self.n_arms = n_arms
        self.d = feature_dim
        self.trials = 0
        self.history_size = history_size
        self.rng = np.random.default_rng(seed)

        # Performance tracking
        self.rewards: Deque[float] = deque(maxlen=history_size)
        self.actions: Deque[int] = deque(maxlen=history_size)

    # Optional exploration rate decay (if applicable).
    def decay_epsilon(self):
        """Optionally decays the exploration rate (epsilon)."""
        pass

    @abstractmethod
    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """Selects an action given the current context.

        Args:
            context (np.ndarray): The current context vector.
            rng (np.random.Generator): Random number generator for exploration.

        Returns:
            int: The index of the selected arm.
        """
        pass

    # Return the current weights/parameters of the agent.
    def get_weights(self) -> Any:
        """Returns the current internal weights or parameters of the agent.

        Returns:
            Any: Agent-specific weights, or None if not applicable.
        """
        return None

    def save(self, path: str) -> None:
        """Saves the agent state to the specified path.

        Args:
            path (str): File system path to save the state.
        """
        pass

    def load(self, path: str) -> None:
        """Loads the agent state from the specified path.

        Args:
            path (str): File system path to load the state from.
        """
        pass

    def reset(self) -> None:
        """Resets the agent state, trials, and history buffers."""
        self.trials = 0
        self.rewards.clear()
        self.actions.clear()
