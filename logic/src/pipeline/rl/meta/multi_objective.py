"""
Multi-Objective Reinforcement Learning (MORL) Weight Optimizer.
"""
import copy
from collections import deque
from typing import Dict, List

import numpy as np

from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy


class ParetoSolution:
    """Represents a solution on the Pareto front."""

    def __init__(self, weights, objectives, reward, model_id=None):
        """
        Initialize a Pareto solution.

        Args:
            weights: Weight configuration dict.
            objectives: Objective values dict.
            reward: Total reward.
            model_id: Optional model identifier.
        """
        self.weights = copy.deepcopy(weights)
        self.objectives = copy.deepcopy(objectives)
        self.reward = reward
        self.model_id = model_id

    def dominates(self, other):
        """
        Check if this solution dominates another.

        Args:
            other: Another ParetoSolution.

        Returns:
            bool: True if this solution dominates the other.
        """
        # Higher is better for all objectives assumed here (should be normalized)
        # Or specifically handled for waste_efficiency (max) and overflow_rate (min)
        waste_better = self.objectives.get("waste_efficiency", 0) >= other.objectives.get("waste_efficiency", 0)
        overflow_better = self.objectives.get("overflow_rate", 1) <= other.objectives.get("overflow_rate", 1)

        strictly_better = (
            self.objectives.get("waste_efficiency", 0) > other.objectives.get("waste_efficiency", 0)
        ) or (self.objectives.get("overflow_rate", 1) < other.objectives.get("overflow_rate", 1))

        return waste_better and overflow_better and strictly_better


class ParetoFront:
    """Maintains a set of non-dominated solutions."""

    def __init__(self, max_size=50):
        """
        Initialize Pareto front.

        Args:
            max_size: Maximum number of solutions to keep.
        """
        self.solutions = []
        self.max_size = max_size

    def add_solution(self, solution):
        """
        Add a solution to the Pareto front if non-dominated.

        Args:
            solution: ParetoSolution to add.

        Returns:
            bool: True if solution was added.
        """
        # Check dominance
        for existing in self.solutions:
            if existing.dominates(solution):
                return False

        # Remove dominated
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        self.solutions.append(solution)

        if len(self.solutions) > self.max_size:
            self.solutions.pop(0)  # Simple pruning for now
        return True


class MORLWeightOptimizer(WeightAdjustmentStrategy):
    """Pareto-based multi-objective weight optimizer."""

    def __init__(
        self,
        initial_weights: Dict[str, float],
        weight_names: List[str] = ["collection", "cost"],
        objective_names: List[str] = ["waste_efficiency", "overflow_rate"],
        history_window: int = 20,
        exploration_factor: float = 0.2,
        adaptation_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initialize MORLWeightOptimizer.

        Args:
            initial_weights: Initial weight configuration.
            weight_names: Names of weight components to optimize.
            objective_names: Names of objectives to track.
            history_window: Window size for performance history.
            exploration_factor: Probability of random exploration.
            adaptation_rate: Rate of weight perturbation.
            **kwargs: Additional keyword arguments.
        """
        self.weight_names = weight_names
        self.objective_names = objective_names
        self.current_weights = copy.deepcopy(initial_weights)
        self.history_window = history_window
        self.performance_history = deque(maxlen=history_window)
        self.exploration_factor = exploration_factor
        self.adaptation_rate = adaptation_rate
        self.pareto_front = ParetoFront()

    def propose_weights(self, context=None):
        """
        Propose weight configuration with optional exploration.

        Args:
            context: Optional context dict (unused).

        Returns:
            Dict[str, float]: Proposed weights.
        """
        if np.random.random() < self.exploration_factor:
            # Random perturbation
            for name in self.weight_names:
                self.current_weights[name] *= 1 + (np.random.random() - 0.5) * self.adaptation_rate
        return self.current_weights

    def _calculate_objectives(self, metrics):
        """Calculate objectives for the Pareto front (compatibility)."""
        den1 = metrics.get("tour_length", metrics.get("cost", 1))
        waste_eff = metrics.get("waste_collected", metrics.get("collection", 0)) / den1 if den1 != 0 else 0

        den2 = metrics.get("total_bins", 1)
        overflow_rate = metrics.get("num_overflows", metrics.get("overflows", 0)) / den2 if den2 != 0 else 0
        return {"waste_efficiency": waste_eff, "overflow_rate": overflow_rate}

    def update_performance_history(self, metrics, reward):
        """Update performance history (compatibility)."""
        self.feedback(reward, metrics)

    def feedback(self, reward, metrics, day=None, step=None):
        """
        Update optimizer with feedback.

        Args:
            reward: Observed reward.
            metrics: Metrics dict with objective values.
            day: Current day (optional).
            step: Current step (optional).
        """
        objectives = self._calculate_objectives(metrics)
        self.performance_history.append(objectives)

        sol = ParetoSolution(self.current_weights, objectives, reward)
        self.pareto_front.add_solution(sol)

    def update_weights(self, metrics=None, reward=None, day=None, step=None):
        """Update weights (compatibility)."""
        if reward is not None and metrics is not None:
            self.feedback(reward, metrics, day, step)
        return self.propose_weights()

    def get_current_weights(self):
        """
        Get current weight configuration.

        Returns:
            Dict[str, float]: Current weights.
        """
        return self.current_weights
