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
        self.weights = copy.deepcopy(weights)
        self.objectives = copy.deepcopy(objectives)
        self.reward = reward
        self.model_id = model_id

    def dominates(self, other):
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
        self.solutions = []
        self.max_size = max_size

    def add_solution(self, solution):
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
        self.weight_names = weight_names
        self.objective_names = objective_names
        self.current_weights = copy.deepcopy(initial_weights)

        self.performance_history = deque(maxlen=history_window)
        self.exploration_factor = exploration_factor
        self.adaptation_rate = adaptation_rate
        self.pareto_front = ParetoFront()

    def propose_weights(self, context=None):
        if np.random.random() < self.exploration_factor:
            # Random perturbation
            for name in self.weight_names:
                self.current_weights[name] *= 1 + (np.random.random() - 0.5) * self.adaptation_rate
        return self.current_weights

    def feedback(self, reward, metrics, day=None, step=None):
        # Simplified objective calculation
        objectives = {
            "waste_efficiency": metrics.get("collection", 0) / (metrics.get("cost", 1) + 1e-6),
            "overflow_rate": metrics.get("overflows", 0),
        }
        self.performance_history.append(objectives)

        sol = ParetoSolution(self.current_weights, objectives, reward)
        self.pareto_front.add_solution(sol)

    def get_current_weights(self):
        return self.current_weights
