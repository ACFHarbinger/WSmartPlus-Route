"""
Contextual Multi-Armed Bandit for Crossover Operator Selection.

This module implements contextual bandits (LinUCB, Thompson Sampling) for
dynamically selecting crossover operators based on population context.

Context Features:
    - Population diversity (measured by genetic distance)
    - Parent fitness similarity
    - Convergence rate (improvement velocity)
    - Iteration phase (early/mid/late)
    - Parent quality (elite vs non-elite)

Algorithms:
    1. **LinUCB (Linear Upper Confidence Bound)** - Li et al., 2010
       - Linear model with UCB exploration bonus
       - Efficient for contextual bandits

    2. **Thompson Sampling (Contextual)** - Agrawal & Goyal, 2013
       - Bayesian approach with posterior sampling
       - Excellent exploration-exploitation balance

    3. **ε-Greedy Contextual** - Baseline
       - Simple ε-greedy with context-aware rewards

Reference:
    Li et al., "A Contextual-Bandit Approach to Personalized News Article Recommendation", WWW 2010.
    Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear Payoffs", ICML 2013.
"""

import contextlib
import random
from collections import defaultdict, deque
from typing import Deque, Dict, List

import numpy as np

from ..hybrid_genetic_search.individual import Individual


class ContextFeatureExtractor:
    """
    Extracts context features from parent individuals and population state.
    """

    def __init__(self, diversity_history_size: int = 10, improvement_history_size: int = 10):
        """Initialize feature extractor."""
        self.diversity_history: Deque[float] = deque(maxlen=diversity_history_size)
        self.improvement_history: Deque[float] = deque(maxlen=improvement_history_size)

    def extract_features(
        self,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
        iteration: int,
        max_iterations: int,
    ) -> np.ndarray:
        """
        Extract context feature vector for crossover operator selection.

        Args:
            p1: First parent.
            p2: Second parent.
            population: Current population.
            iteration: Current iteration number.
            max_iterations: Maximum iterations.

        Returns:
            Feature vector (normalized to [0, 1]).
        """
        features = []

        # Feature 1: Parent fitness similarity (0 = identical, 1 = very different)
        if population:
            fitness_range = max(ind.profit_score for ind in population) - min(ind.profit_score for ind in population)
            fitness_diff = abs(p1.profit_score - p2.profit_score) / fitness_range if fitness_range > 1e-06 else 0.0
        else:
            fitness_diff = 0.5
        features.append(min(1.0, fitness_diff))

        # Feature 2: Parent genetic distance (Hamming distance on giant tour)
        if len(p1.giant_tour) > 0 and len(p2.giant_tour) > 0:
            common_nodes = set(p1.giant_tour) & set(p2.giant_tour)
            total_nodes = len(set(p1.giant_tour) | set(p2.giant_tour))
            genetic_distance = 1.0 - len(common_nodes) / max(total_nodes, 1)
        else:
            genetic_distance = 0.5
        features.append(genetic_distance)

        # Feature 3: Population diversity (average distance between individuals)
        diversity = self._calculate_population_diversity(population)
        self.diversity_history.append(diversity)
        features.append(diversity)

        # Feature 4: Iteration phase (0 = early, 1 = late)
        iteration_phase = iteration / max(max_iterations, 1)
        features.append(iteration_phase)

        # Feature 5: Parent quality (average of parent fitness ranks)
        if population:
            sorted_pop = sorted(population, key=lambda x: x.profit_score, reverse=True)
            rank_p1 = next((i for i, ind in enumerate(sorted_pop) if ind is p1), len(sorted_pop))
            rank_p2 = next((i for i, ind in enumerate(sorted_pop) if ind is p2), len(sorted_pop))
            avg_rank = (rank_p1 + rank_p2) / (2 * max(len(sorted_pop), 1))
        else:
            avg_rank = 0.5
        features.append(avg_rank)

        # Feature 6: Convergence rate (recent improvement velocity)
        convergence_rate = np.mean(self.improvement_history) if self.improvement_history else 0.5
        features.append(min(1.0, convergence_rate))

        # Feature 7: Parent diversity contribution
        p1_diversity = self._individual_diversity(p1, population)
        p2_diversity = self._individual_diversity(p2, population)
        avg_parent_diversity = (p1_diversity + p2_diversity) / 2.0
        features.append(avg_parent_diversity)

        # Feature 8: Bias term (always 1.0 for linear models)
        features.append(1.0)

        return np.array(features, dtype=np.float64)

    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """Calculate average genetic diversity in population."""
        if len(population) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                s1 = set(population[i].giant_tour)
                s2 = set(population[j].giant_tour)
                distance = len(s1.symmetric_difference(s2)) / max(len(s1 | s2), 1)
                total_distance += distance
                count += 1

        return total_distance / max(count, 1)

    def _individual_diversity(self, ind: Individual, population: List[Individual]) -> float:
        """Calculate average distance of individual to population."""
        if len(population) <= 1:
            return 0.0

        total_distance = 0.0
        count = 0

        s1 = set(ind.giant_tour)
        for other in population:
            if other is not ind:
                s2 = set(other.giant_tour)
                distance = len(s1.symmetric_difference(s2)) / max(len(s1 | s2), 1)
                total_distance += distance
                count += 1

        return total_distance / max(count, 1)

    def update_improvement(self, improvement: float):
        """Update improvement history."""
        self.improvement_history.append(improvement)


class LinUCBCrossoverSelector:
    """
    Linear Upper Confidence Bound (LinUCB) for crossover operator selection.

    Maintains a linear model for each operator's expected reward given context.
    """

    def __init__(
        self,
        n_operators: int,
        feature_dim: int = 8,
        alpha: float = 1.0,  # Exploration parameter
        operator_reward_size: int = 50,
        operator_selection_threshold: float = 1e-9,
        improvement_threshold: float = 1e-6,
    ):
        """
        Initialize LinUCB selector.

        Args:
            n_operators: Number of crossover operators.
            feature_dim: Dimension of context feature vector.
            alpha: Exploration parameter (UCB width).
            operator_reward_size: Size of operator reward history.
            operator_selection_threshold: Threshold for operator selection.
            improvement_threshold: Threshold for improvement detection.
        """
        self.n_operators = n_operators
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.operator_selection_threshold = operator_selection_threshold
        self.improvement_threshold = improvement_threshold

        # For each operator, maintain:
        # A: design matrix (feature outer products)
        # b: response vector (feature × reward)
        self.A = [np.identity(feature_dim) for _ in range(n_operators)]
        self.b = [np.zeros(feature_dim) for _ in range(n_operators)]

        # Tracking
        self.operator_uses: Dict[int, int] = defaultdict(int)
        self.operator_rewards: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=operator_reward_size))

    def select_operator(self, context: np.ndarray, rng: random.Random) -> int:
        """
        Select operator using LinUCB policy.

        Args:
            context: Context feature vector.
            rng: Random number generator.

        Returns:
            Selected operator index.
        """
        ucb_values = []

        for op_idx in range(self.n_operators):
            # Compute weight vector: θ = A^(-1) * b
            A_inv = np.linalg.inv(self.A[op_idx])
            theta = A_inv @ self.b[op_idx]

            # Expected reward: θ^T * x
            expected_reward = theta @ context

            # UCB bonus: α * sqrt(x^T * A^(-1) * x)
            ucb_bonus = self.alpha * np.sqrt(context @ A_inv @ context)

            # UCB value
            ucb = expected_reward + ucb_bonus
            ucb_values.append(ucb)

        # Select operator with highest UCB
        max_ucb = max(ucb_values)
        best_operators = [i for i, v in enumerate(ucb_values) if abs(v - max_ucb) < self.improvement_threshold]

        return rng.choice(best_operators)

    def update(self, operator: int, context: np.ndarray, reward: float):
        """
        Update LinUCB model with observed reward.

        Args:
            operator: Selected operator index.
            context: Context feature vector.
            reward: Observed reward.
        """
        # Update design matrix: A = A + x * x^T
        self.A[operator] += np.outer(context, context)

        # Update response vector: b = b + r * x
        self.b[operator] += reward * context

        # Track performance
        self.operator_uses[operator] += 1
        self.operator_rewards[operator].append(reward)

    def get_statistics(self) -> Dict:
        """Get operator selection statistics."""
        stats = {
            "total_uses": sum(self.operator_uses.values()),
            "operator_uses": dict(self.operator_uses),
            "operator_avg_rewards": {},
        }

        for op_idx, rewards in self.operator_rewards.items():
            if rewards:
                stats["operator_avg_rewards"][op_idx] = float(np.mean(rewards))

        return stats


class ThompsonSamplingCrossoverSelector:
    """
    Thompson Sampling with linear Gaussian model for crossover selection.

    Uses Bayesian linear regression with posterior sampling.
    """

    def __init__(
        self,
        n_operators: int,
        feature_dim: int = 8,
        lambda_prior: float = 1.0,  # Prior precision
        noise_variance: float = 0.1,  # Observation noise
        improvement_threshold: float = 1e-9,
        operator_reward_size: int = 50,
    ):
        """
        Initialize Thompson Sampling selector.

        Args:
            n_operators: Number of crossover operators.
            feature_dim: Dimension of context feature vector.
            lambda_prior: Prior precision for weights.
            noise_variance: Assumed noise variance in rewards.
            improvement_threshold: Threshold for improvement detection.
            operator_reward_size: Size of operator reward history.
        """
        self.n_operators = n_operators
        self.feature_dim = feature_dim
        self.lambda_prior = lambda_prior
        self.noise_variance = noise_variance
        self.improvement_threshold = improvement_threshold

        # For each operator, maintain:
        # B: precision matrix (inverse covariance)
        # μ: mean weight vector
        self.B = [lambda_prior * np.identity(feature_dim) for _ in range(n_operators)]
        self.mu = [np.zeros(feature_dim) for _ in range(n_operators)]
        self.f = [np.zeros(feature_dim) for _ in range(n_operators)]  # μ_tilde

        # Tracking
        self.operator_uses: Dict[int, int] = defaultdict(int)
        self.operator_rewards: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=operator_reward_size))

    def select_operator(self, context: np.ndarray, rng: random.Random) -> int:
        """
        Select operator using Thompson Sampling.

        Args:
            context: Context feature vector.
            rng: Random number generator (note: uses numpy's random for sampling).

        Returns:
            Selected operator index.
        """
        sampled_rewards = []

        for op_idx in range(self.n_operators):
            # Sample θ from posterior N(μ, B^(-1))
            try:
                B_inv = np.linalg.inv(self.B[op_idx])
                # Sample from multivariate normal
                theta_sample = np.random.multivariate_normal(self.mu[op_idx], B_inv)
            except np.linalg.LinAlgError:
                # Fallback to mean if matrix is singular
                theta_sample = self.mu[op_idx]

            # Compute expected reward with sampled θ
            sampled_reward = theta_sample @ context
            sampled_rewards.append(sampled_reward)

        # Select operator with highest sampled reward
        max_reward = max(sampled_rewards)
        best_operators = [i for i, v in enumerate(sampled_rewards) if abs(v - max_reward) < self.improvement_threshold]

        return rng.choice(best_operators)

    def update(self, operator: int, context: np.ndarray, reward: float):
        """
        Update Thompson Sampling model with observed reward.

        Args:
            operator: Selected operator index.
            context: Context feature vector.
            reward: Observed reward.
        """
        # Update precision matrix: B = B + (1/σ²) * x * x^T
        self.B[operator] += (1.0 / self.noise_variance) * np.outer(context, context)

        # Update f: f = f + (1/σ²) * r * x
        self.f[operator] += (1.0 / self.noise_variance) * reward * context

        # Update mean: μ = B^(-1) * f
        with contextlib.suppress(np.linalg.LinAlgError):
            self.mu[operator] = np.linalg.inv(self.B[operator]) @ self.f[operator]

        # Track performance
        self.operator_uses[operator] += 1
        self.operator_rewards[operator].append(reward)

    def get_statistics(self) -> Dict:
        """Get operator selection statistics."""
        stats = {
            "total_uses": sum(self.operator_uses.values()),
            "operator_uses": dict(self.operator_uses),
            "operator_avg_rewards": {},
        }

        for op_idx, rewards in self.operator_rewards.items():
            if rewards:
                stats["operator_avg_rewards"][op_idx] = float(np.mean(rewards))

        return stats


class EpsilonGreedyCrossoverSelector:
    """
    ε-Greedy with context-aware Q-values (baseline).

    Simpler alternative to LinUCB and Thompson Sampling.
    """

    def __init__(
        self,
        n_operators: int,
        alpha: float = 0.1,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        operator_reward_size: int = 50,
    ):
        """
        Initialize ε-Greedy selector.

        Args:
            n_operators: Number of crossover operators.
            alpha: Learning rate.
            epsilon: Initial exploration rate.
            epsilon_decay: Decay factor for epsilon.
            epsilon_min: Minimum epsilon value.
            operator_reward_size: Size of operator reward history.
        """
        self.n_operators = n_operators
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Simple Q-values per operator (not context-dependent)
        self.q_values = np.zeros(n_operators)
        self.operator_uses: Dict[int, int] = defaultdict(int)
        self.operator_rewards: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=operator_reward_size))

    def select_operator(self, context: np.ndarray, rng: random.Random) -> int:
        """
        Select operator using ε-greedy policy.

        Args:
            context: Context feature vector (not used in basic ε-greedy).
            rng: Random number generator.

        Returns:
            Selected operator index.
        """
        if rng.random() < self.epsilon:
            # Explore: random operator
            return rng.randint(0, self.n_operators - 1)
        else:
            # Exploit: best operator
            max_q = np.max(self.q_values)
            best_operators = np.where(self.q_values == max_q)[0]
            return rng.choice(best_operators)

    def update(self, operator: int, context: np.ndarray, reward: float):
        """
        Update Q-value with observed reward.

        Args:
            operator: Selected operator index.
            context: Context feature vector (not used).
            reward: Observed reward.
        """
        self.q_values[operator] = (1 - self.alpha) * self.q_values[operator] + self.alpha * reward

        # Track performance
        self.operator_uses[operator] += 1
        self.operator_rewards[operator].append(reward)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_statistics(self) -> Dict:
        """Get operator selection statistics."""
        stats = {
            "epsilon": self.epsilon,
            "total_uses": sum(self.operator_uses.values()),
            "operator_uses": dict(self.operator_uses),
            "operator_avg_rewards": {},
            "q_values": self.q_values.tolist(),
        }

        for op_idx, rewards in self.operator_rewards.items():
            if rewards:
                stats["operator_avg_rewards"][op_idx] = float(np.mean(rewards))

        return stats
