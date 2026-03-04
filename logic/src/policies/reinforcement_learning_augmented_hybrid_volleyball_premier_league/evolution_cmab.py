"""
Enhanced HGS Evolution with Contextual Multi-Armed Bandits.

This module extends the standard HGS evolution operators with intelligent
crossover operator selection using contextual bandits.

Features:
    - LinUCB-based operator selection (default)
    - Thompson Sampling alternative
    - ε-Greedy baseline
    - Context-aware reward calculation
    - Adaptive operator portfolio

Reference:
    Vidal et al., "A hybrid genetic algorithm for multidepot and periodic VRP", 2012.
    Li et al., "Contextual Bandits for Adaptive Operator Selection", WWW 2010.
"""

import random
from typing import List, Optional

import numpy as np

from ..hybrid_genetic_search.individual import Individual
from ..hybrid_genetic_search.split import LinearSplit
from .contextual_mab import (
    ContextFeatureExtractor,
    EpsilonGreedyCrossoverSelector,
    LinUCBCrossoverSelector,
    ThompsonSamplingCrossoverSelector,
)
from .crossover_operators import CROSSOVER_NAMES, CROSSOVER_OPERATORS


class CMABEvolution:
    """
    Contextual Multi-Armed Bandit based evolutionary operators.

    Dynamically selects crossover operators based on population context.
    """

    def __init__(
        self,
        split_manager: LinearSplit,
        bandit_algorithm: str = "linucb",  # 'linucb', 'thompson', 'epsilon_greedy'
        max_iterations: int = 1000,
        quality_weight: float = 0.5,
        improvement_weight: float = 0.6,
        diversity_weight: float = 0.2,
        novelty_weight: float = 1.0,
        reward_threshold: float = 1e-6,
        default_reward: float = 5.0,
        rng: Optional[random.Random] = None,
        **kwargs,
    ):
        """
        Initialize CMAB evolution.

        Args:
            split_manager: LinearSplit for evaluating individuals.
            bandit_algorithm: Bandit algorithm to use.
            max_iterations: Maximum number of iterations.
            quality_weight: Weight for quality component.
            improvement_weight: Weight for improvement component.
            diversity_weight: Weight for diversity component.
            novelty_weight: Weight for novelty component.
            reward_threshold: Threshold for reward calculation.
            default_reward: Default reward value.
            rng: Random number generator.
            **kwargs: Additional parameters for bandit algorithm.
        """
        self.split_manager = split_manager
        self.rng = rng if rng is not None else random.Random()

        # Initialize feature extractor
        self.feature_extractor = ContextFeatureExtractor(
            diversity_history_size=kwargs.get("diversity_history_size", 10),
            improvement_history_size=kwargs.get("improvement_history_size", 10),
        )

        # Initialize bandit selector
        n_operators = len(CROSSOVER_OPERATORS)
        if bandit_algorithm == "linucb":
            self.bandit = LinUCBCrossoverSelector(
                n_operators,
                feature_dim=kwargs.get("feature_dim", 8),
                alpha=kwargs.get("alpha", 1.0),
                operator_reward_size=kwargs.get("operator_reward_size", 50),
                operator_selection_threshold=kwargs.get("operator_selection_threshold", 1e-9),
                improvement_threshold=kwargs.get("improvement_threshold", 1e-6),
            )
        elif bandit_algorithm == "thompson":
            self.bandit = ThompsonSamplingCrossoverSelector(
                n_operators,
                feature_dim=kwargs.get("feature_dim", 8),
                lambda_prior=kwargs.get("lambda_prior", 1.0),
                noise_variance=kwargs.get("noise_variance", 0.1),
                operator_reward_size=kwargs.get("operator_reward_size", 50),
            )
        elif bandit_algorithm == "epsilon_greedy":
            self.bandit = EpsilonGreedyCrossoverSelector(
                n_operators,
                alpha=kwargs.get("alpha", 0.1),
                epsilon=kwargs.get("epsilon", 0.1),
                epsilon_decay=kwargs.get("epsilon_decay", 0.999),
                epsilon_min=kwargs.get("epsilon_min", 0.01),
                operator_reward_size=kwargs.get("operator_reward_size", 50),
            )
        else:
            raise ValueError(f"Unknown bandit algorithm: {bandit_algorithm}")

        self.bandit_algorithm = bandit_algorithm
        self.quality_weight = quality_weight
        self.improvement_weight = improvement_weight
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.reward_threshold = reward_threshold
        self.default_reward = default_reward

        # Tracking
        self.iteration = 0
        self.max_iterations = max_iterations  # Updated during evolution

    def crossover(
        self,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
        iteration: int,
        max_iterations: int,
    ) -> Individual:
        """
        Perform crossover with CMAB operator selection.

        Args:
            p1: First parent.
            p2: Second parent.
            population: Current population.
            iteration: Current iteration number.
            max_iterations: Maximum iterations.

        Returns:
            Child individual.
        """
        self.iteration = iteration
        self.max_iterations = max_iterations

        # Extract context features
        context = self.feature_extractor.extract_features(p1, p2, population, iteration, max_iterations)

        # Select crossover operator using bandit
        op_idx = self.bandit.select_operator(context, self.rng)
        operator_name = CROSSOVER_NAMES[op_idx]
        operator_func = CROSSOVER_OPERATORS[operator_name]

        # Apply crossover
        child = operator_func(p1, p2, self.rng)

        # Evaluate child
        self.evaluate(child)

        # Calculate reward based on child quality
        reward = self._calculate_reward(child, p1, p2, population)

        # Update bandit
        self.bandit.update(op_idx, context, reward)

        # Store operator info for visualization
        child.crossover_operator = operator_name
        child.crossover_reward = reward

        return child

    def _calculate_reward(
        self,
        child: Individual,
        p1: Individual,
        p2: Individual,
        population: List[Individual],
    ) -> float:
        """
        Calculate reward for the crossover operator.

        Reward combines:
            1. Absolute quality (profit score)
            2. Improvement over parents
            3. Diversity contribution
            4. Novelty contribution

        Args:
            child: Offspring individual.
            p1: First parent.
            p2: Second parent.
            population: Current population.

        Returns:
            Reward value (higher is better).
        """
        if not population:
            return self.default_reward

        # 1. Component: Elite Improvement (The 'Leapfrog' effect)
        # Reward based on how much the child improves over the BETTER parent.
        best_parent_profit = max(p1.profit_score, p2.profit_score)

        # Normalized improvement
        improvement = (
            (child.profit_score - best_parent_profit) / abs(best_parent_profit)
            if abs(best_parent_profit) > self.reward_threshold
            else 0.0
        )

        # Scale: Significant reward only for actual improvement
        # We use a non-linear scaling here to heavily reward "Elite" discoveries
        improvement_reward = np.sign(improvement) * np.log1p(abs(improvement * 100))

        # 2. Component: Global Progress
        # How does this child compare to the current global best?
        global_best = max(ind.profit_score for ind in population)
        global_worst = min(ind.profit_score for ind in population)

        profit_range = global_best - global_worst
        quality_reward = (
            (child.profit_score - global_worst) / profit_range
            if profit_range > self.reward_threshold
            else self.default_reward / 10
        )

        # 3. Component: Diversity Credit
        # Crucial for HGS to prevent "Inbreeding"
        child_diversity = self.feature_extractor._individual_diversity(child, population)

        # 4. Component: Penalty for Stagnation
        # If the child is exactly the same as a parent (no novelty), penalize.
        novelty_penalty = 0.0
        if child.profit_score == p1.profit_score or child.profit_score == p2.profit_score:
            novelty_penalty = -self.default_reward / 10

        # Weighted Sum of Rewards
        # We increase the weight of improvement_reward as it's the hardest to achieve
        total_reward = (
            (self.quality_weight * quality_reward)
            + (self.improvement_weight * improvement_reward)
            + (self.diversity_weight * child_diversity)
            + (self.novelty_weight * novelty_penalty)
        )

        # 5. Output: Tanh Soft-Clipping to [0, 10]
        # This ensures that even massive improvements don't cause Q-value explosion
        # while keeping the gradient sensitive near the 5.0 (neutral) mark.
        return self.default_reward * (np.tanh(total_reward) + 1.0)

    def evaluate(self, ind: Individual):
        """
        Evaluate individual using LinearSplit.

        Args:
            ind: Individual to evaluate.
        """
        routes, profit = self.split_manager.split(ind.giant_tour)
        ind.routes = routes
        ind.profit_score = profit

        # Calculate cost/revenue
        rev = 0.0
        cost = 0.0
        for r in routes:
            if not r:
                continue
            d = self.split_manager.dist_matrix[0, r[0]]
            rev += self.split_manager.wastes[r[0]] * self.split_manager.R
            for i in range(len(r) - 1):
                d += self.split_manager.dist_matrix[r[i], r[i + 1]]
                rev += self.split_manager.wastes[r[i + 1]] * self.split_manager.R
            d += self.split_manager.dist_matrix[r[-1], 0]
            cost += d * self.split_manager.C

        ind.cost = cost
        ind.revenue = rev

    def update_improvement(self, improvement: float):
        """
        Update feature extractor with improvement info.

        Args:
            improvement: Population improvement rate.
        """
        self.feature_extractor.update_improvement(improvement)

    def decay_exploration(self):
        """Decay exploration rate (for ε-greedy)."""
        if hasattr(self.bandit, "decay_epsilon"):
            self.bandit.decay_epsilon()

    def get_statistics(self) -> dict:
        """
        Get CMAB statistics for visualization.

        Returns:
            Dictionary of statistics.
        """
        stats = self.bandit.get_statistics()
        stats["bandit_algorithm"] = self.bandit_algorithm
        stats["iteration"] = self.iteration

        # Add operator names
        stats["operator_names"] = CROSSOVER_NAMES

        return stats


def update_biased_fitness(
    population: List[Individual],
    nb_elite: int,
    alpha_diversity: float = 0.5,
    neighbor_size: int = 15,
):
    """
    Update biased fitness based on profit rank and diversity rank.

    This is the same as the original HGS, maintained for compatibility.

    Args:
        population: List of individuals to update.
        nb_elite: Number of elite individuals to protect.
        alpha_diversity: Weight for diversity in fitness calculation.
        neighbor_size: Number of nearest neighbors to consider.
    """
    if not population:
        return
    pop_size = len(population)
    if pop_size == 0:
        return

    # Rank by Profit
    population.sort(key=lambda x: x.profit_score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    # Diversity: Average distance to closest individuals based on VISITED node sets
    for i, ind1 in enumerate(population):
        dists = []
        s1 = set(n for r in ind1.routes for n in r)
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            s2 = set(n for r in ind2.routes for n in r)
            # Hamming-like distance on set of visited nodes
            dist = len(s1.symmetric_difference(s2))
            dists.append(dist)
        dists.sort()
        # Avg of closest neighbor_size
        ind1.dist_to_parents = float(sum(dists[:neighbor_size]) / neighbor_size) if dists else 0.0

    # Rank by Diversity
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Biased Fitness = Rank(Profit) + alpha_diversity * Rank(Diversity)
    # ELITE PROTECTION: Ensure top nb_elite profit solutions always survive.
    for ind in population:
        if ind.rank_profit <= nb_elite:
            # Pure profit ranking for top elites ensures their survival in trim
            ind.fitness = float(ind.rank_profit)
        else:
            # Biased fitness for the rest to maintain diversity
            ind.fitness = ind.rank_profit + alpha_diversity * ind.rank_diversity
