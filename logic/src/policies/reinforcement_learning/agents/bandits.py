from typing import Any, Dict, Optional

import numpy as np

from .base import RLAgent


class BanditAgent(RLAgent):
    """
    Base class for Multi-Armed Bandit (MAB) agents.

    Maintains basic statistics for each arm (action) including selection counts
    and estimated average rewards. Provides core mechanisms for state persistence
    and diagnostic reporting.

    Attributes:
        n_arms: The number of available arms (actions).
        counts: NumPy array of selection counts for each arm.
        values: NumPy array of current reward estimates for each arm.
        rng: Random number generator for deterministic stability.
        reward_history: Dictionary mapping arm indices to deques of recent rewards.
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the bandit agent.

        Args:
            n_arms: Number of available actions.
            seed: Optional seed for the random number generator.
            history_size: Maximum size of the reward tracking buffer for each arm.
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.rng = np.random.default_rng(seed)

        # Performance tracking
        from collections import deque

        self.reward_history = {i: deque(maxlen=history_size) for i in range(n_arms)}

    def decay_epsilon(self) -> None:
        """
        Optional epsilon decay mechanism.

        Implemented by classes that utilize randomized exploration.
        """
        pass

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def save(self, path: str) -> None:
        """Save the agent's state to a file using pickle."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """Load the agent's state from a file."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic statistics about the agent.

        Returns:
            Dictionary containing selection counts, values, and the best arm.
        """
        return {
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "best_arm": int(np.argmax(self.values)) if self.n_arms > 0 else None,
            "avg_history_rewards": {i: float(np.mean(h)) if h else 0.0 for i, h in self.reward_history.items()},
        }

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Standard incremental update for stationary environments.

        Formula: Q(a) = Q(a) + (1/n) * [r - Q(a)]

        Args:
            state: Initial state (unused).
            action: The selected arm index.
            reward: Observed reward.
            next_state: Resulting state (unused).
            done: Termination flag (unused).
        """
        # Track reward history
        self.reward_history[action].append(reward)

        # Incremental update
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n


class EpsilonGreedyBandit(BanditAgent):
    """
    Epsilon-Greedy Bandit policy.

    Balances exploration and exploitation by selecting a random arm with
    probability 'epsilon' and the best known arm with probability '1 - epsilon'.
    Supports dynamic epsilon decay to favor exploitation as search progresses.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Epsilon-Greedy agent.

        Args:
            n_arms: Number of available actions.
            epsilon: Initial exploration probability.
            epsilon_decay: Multiplicative factor for reducing epsilon.
            epsilon_min: Minimum allowable exploration probability.
            seed: Optional seed for the random number generator.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def decay_epsilon(self) -> None:
        """Apply multiplicative decay to the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the epsilon-greedy policy.

        Args:
            state: Contextual state (unused by non-contextual bandits).
            rng: Optional RNG for local selection logic.

        Returns:
            The selected arm index.
        """
        rng = rng or self.rng

        # Exploration: Select a random arm with probability epsilon
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_arms))

        # Exploitation: Select the arm with the highest estimated value
        return int(np.argmax(self.values))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment basic statistics with current epsilon.

        Returns:
            Dictionary including basic stats and current exploration rate.
        """
        stats = super().get_statistics()
        stats["epsilon"] = self.epsilon
        return stats


class UCBBandit(BanditAgent):
    """
    Upper Confidence Bound (UCB1) Bandit policy.

    The UCB1 algorithm implements 'optimism in the face of uncertainty' by
    adding a confidence bonus to the estimated value of each arm. This bonus
    decreases as the arm is selected more frequently.
    """

    def __init__(self, n_arms: int, c: float = 2.0, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the UCB1 agent.

        Args:
            n_arms: Number of available actions.
            c: Exploration parameter (controls the width of the confidence interval).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.c = c

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the UCB1 policy.

        Rules:
            1. Select each arm at least once (bootstrap).
            2. For subsequent trials, maximize V_j + c * sqrt(ln(total_trials) / counts_j).

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        # Ensure every arm is explored at least once
        if 0 in self.counts:
            return int(np.where(self.counts == 0)[0][0])

        # Calculate UCB values for all arms
        total_counts = np.sum(self.counts)
        # Confidence bonus: bonus increases with total trials, decreases with arm-specific trials
        bonus = self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-12))
        ucb_values = self.values + bonus

        return int(np.argmax(ucb_values))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with exploration constant 'c'.

        Returns:
            Dictionary including basic stats and UCB parameter.
        """
        stats = super().get_statistics()
        stats["c"] = self.c
        return stats


class SoftmaxBandit(BanditAgent):
    """
    Softmax Sampling Bandit policy.

    Selects arms with a probability proportional to their estimated values using
    the Gibbs (Boltzmann) distribution. Controlled by a 'temperature' parameter.
    """

    def __init__(self, n_arms: int, temperature: float = 1.0, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the Softmax agent.

        Args:
            n_arms: Number of available actions.
            temperature: Exploration temperature (higher = uniform, lower = greedy).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.temperature = temperature
        self.probs = np.ones(n_arms) / n_arms

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using Softmax sampling.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        rng = rng or self.rng

        # Calculate exponentiated values for probability distribution
        # clip values to prevent overflow in exp
        v = np.clip(self.values / max(self.temperature, 1e-8), -100, 100)
        exp_v = np.exp(v)

        # Normalize to generate probabilities
        self.probs = exp_v / np.sum(exp_v)

        return int(rng.choice(self.n_arms, p=self.probs))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with temperature and selection probabilities.

        Returns:
            Dictionary containing model parameters and current distributions.
        """
        stats = super().get_statistics()
        stats.update({"temperature": self.temperature, "probs": self.probs.tolist()})
        return stats


class ThompsonSamplingBandit(BanditAgent):
    """
    Thompson Sampling Bandit policy for Bernoulli environments.

    Uses Beta distribution priors for each arm's success probability.
    Selects actions by sampling from the posterior and playing the arm
    with the highest sampled success rate.
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Thompson Sampling agent.

        Args:
            n_arms: Number of available actions.
            alpha_prior: Initial success count for Beta distribution.
            beta_prior: Initial failure count for Beta distribution.
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.alphas = np.full(n_arms, alpha_prior, dtype=float)
        self.betas = np.full(n_arms, beta_prior, dtype=float)

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using Thompson sampling.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        rng = rng or self.rng

        # Sample from the Beta distribution of each arm
        samples = [rng.beta(self.alphas[i], self.betas[i]) for i in range(self.n_arms)]

        # Select the arm with the highest sample
        return int(np.argmax(samples))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the Beta posterior for the selected arm.

        Args:
            state: Initial state.
            action: The arm that was played.
            reward: Observed binary reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Standard bandit update for counts/values
        super().update(state, action, reward, next_state, done)

        # Update Beta parameters based on reward (assumes binary reward)
        # Success count (alpha) increases with reward, failure (beta) increases otherwise
        if reward > 0.5:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1

    def reset(self) -> None:
        """Reset internal parameters and success/failure counts."""
        super().reset()
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with Beta distribution parameters.

        Returns:
            Dictionary including basic stats and posterior alpha/beta values.
        """
        stats = super().get_statistics()
        stats.update({"alphas": self.alphas.tolist(), "betas": self.betas.tolist()})
        return stats


class DiscountedUCBBandit(BanditAgent):
    """
    Discounted UCB Bandit policy.

    Useful for non-stationary environments where the optimal action may change
    over time. Uses a discount factor 'gamma' to slowly decay the contribution
    of past observations.
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        gamma: float = 0.95,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Discounted UCB agent.

        Args:
            n_arms: Number of available actions.
            c: Exploration parameter.
            gamma: Discount factor (0 < gamma <= 1).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.c = c
        self.gamma = gamma
        self.discounted_counts = np.zeros(n_arms)
        self.discounted_rewards = np.zeros(n_arms)

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the Discounted UCB policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        # Bootstrap: ensure every arm has at least some discounted representation
        if np.any(self.discounted_counts < 1.0):
            return int(np.argmin(self.discounted_counts))

        # Calculate UCB values using discounted statistics
        total_discounted = np.sum(self.discounted_counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(total_discounted) / (self.discounted_counts + 1e-9))

        return int(np.argmax(ucb_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update discounted statistics for all arms.

        Args:
            state: Initial state.
            action: Selected arm.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Apply discount to all arm stats
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma

        # Increment stats for the selected arm
        self.discounted_counts[action] += 1
        self.discounted_rewards[action] += reward

        # Update current value estimate using discounted mean
        self.values[action] = self.discounted_rewards[action] / self.discounted_counts[action]

        # Use super().update if tracking of raw counts/history is desired
        # but avoid the incremental mean update which is inappropriate here
        self.reward_history[action].append(reward)
        self.counts[action] += 1


class SlidingWindowUCBBandit(BanditAgent):
    """
    Sliding Window UCB Bandit policy.

    Handles volatile environments by selecting arms based only on the most
    recent 'window_size' observations.
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 100,
        c: float = 2.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Sliding Window UCB agent.

        Args:
            n_arms: Number of available actions.
            window_size: Number of past observations to consider.
            c: Exploration parameter.
            seed: Optional seed for the RNG.
            history_size: Size of diagnostic reward history.
        """
        super().__init__(n_arms, seed, history_size)
        self.window_size = window_size
        self.c = c
        from collections import deque

        # Maintain window-specific reward deques
        self.windows = [deque(maxlen=window_size) for _ in range(n_arms)]

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the Sliding Window UCB policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        counts = [len(w) for w in self.windows]

        # Ensure every arm has at least one observation in its current window
        if 0 in counts:
            return int(counts.index(0))

        # Calculate UCB values based only on the counts and values within the window
        total = sum(counts)
        ucb_values = [
            self.values[i] + self.c * np.sqrt(np.log(total) / len(self.windows[i])) for i in range(self.n_arms)
        ]

        return int(np.argmax(ucb_values))

    def reset(self) -> None:
        """Clear all windows and reset stats."""
        from collections import deque

        super().reset()
        self.windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Add observed reward to the window and update the mean estimate.

        Args:
            state: Initial state.
            action: Selected arm index.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Update sliding window
        self.windows[action].append(reward)

        # Update value using the mean of the current window
        self.values[action] = np.mean(self.windows[action])

        # Track raw statistics
        self.counts[action] += 1
        self.reward_history[action].append(reward)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with the current contents of the sliding windows.

        Returns:
            Dictionary including basic stats and the rewards in each window.
        """
        stats = super().get_statistics()
        stats.update({"windows": [list(w) for w in self.windows]})
        return stats


class EXP3Agent(BanditAgent):
    """
    EXP3 algorithm for adversarial bandits.

    Exponential-weight algorithm for Exploration and Exploitation. Designed to
    handle adversarial environments where the rewards are not drawn from a
    fixed distribution.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the EXP3 agent.

        Args:
            n_arms: Number of available actions.
            gamma: Exploration parameter (controls the mixture between weights and uniform dist).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.probs = np.ones(n_arms) / n_arms

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the EXP3 policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        rng = rng or self.rng

        # Calculate probabilities based on current weights
        total_w = np.sum(self.weights)
        # Prob = (1-gamma) * (weight / total_weight) + (gamma / n_arms)
        self.probs = (1 - self.gamma) * (self.weights / (total_w + 1e-12)) + (self.gamma / self.n_arms)

        # Sample according to probabilities
        return int(rng.choice(self.n_arms, p=self.probs))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the weights of the selected arm.

        Args:
            state: Initial state.
            action: Selected arm index.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Standard tracking
        self.reward_history[action].append(reward)
        self.counts[action] += 1

        # Scale reward to [0, 1] for stable exponential updates
        rew = np.clip((reward + 10) / 20, 0, 1)

        # Importance-weighted reward estimate: e_r = r / prob(action)
        est_reward = rew / (self.probs[action] + 1e-12)

        # Update weight: w = w * exp(gamma * est_reward / n_arms)
        self.weights[action] *= np.exp(self.gamma * est_reward / self.n_arms)

    def reset(self) -> None:
        """Reset weights and probabilities to uniform distribution."""
        super().reset()
        self.weights = np.ones(self.n_arms)
        self.probs = np.ones(self.n_arms) / self.n_arms

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with EXP3 specific weights and probabilities.

        Returns:
            Dictionary including basic stats, weights, and current probabilities.
        """
        stats = super().get_statistics()
        stats.update({"weights": self.weights.tolist(), "probs": self.probs.tolist()})
        return stats
