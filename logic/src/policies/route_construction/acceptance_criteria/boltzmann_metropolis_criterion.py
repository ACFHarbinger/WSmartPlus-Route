"""
Boltzmann-Metropolis Criterion (BMC) / Simulated Annealing.
"""

import math
import random
from typing import Any, Dict, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class BoltzmannAcceptance(IAcceptanceCriterion):
    """
    Metropolis-Boltzmann acceptance criterion governed by a geometric cooling schedule.

    Improving moves are accepted deterministically. Worsening moves are accepted
    with probability P = exp(Δf / T), where Δf < 0 for maximization.
    """

    def __init__(self, initial_temp: float, alpha: float, seed: Optional[int] = 42):
        """
        Args:
            initial_temp (float): The starting temperature parameter.
            alpha (float): The geometric cooling rate (typically 0.90 - 0.995).
            seed (Optional[int]): Random seed for stochastic reproducibility.
        """
        self.T = initial_temp
        self.alpha = alpha
        self.rng = random.Random(seed)

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float) -> bool:
        delta = candidate_obj - current_obj
        if delta >= 0:
            return True

        # delta is negative, T is positive -> exp(negative) gives valid [0, 1] probability
        if self.T <= 1e-9:
            return False

        return self.rng.random() < math.exp(delta / self.T)

    def step(self, current_obj: float, candidate_obj: float, accepted: bool) -> None:
        # Geometric decay
        self.T *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        return {"temperature": self.T}
