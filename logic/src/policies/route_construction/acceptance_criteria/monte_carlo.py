"""
Monte Carlo Acceptance (MCA) Criterion.
"""

import random
from typing import Any, Dict, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class MonteCarloAcceptance(IAcceptanceCriterion):
    """
    Fixed-probability stochastic acceptance.
    Deterministically accepts improving candidates.
    Worsening candidates are accepted with a fixed, unannealed probability `p`.
    """

    def __init__(self, p: float = 0.1, seed: Optional[int] = None) -> None:
        """
        Args:
            p (float): The probability of accepting a worsening move (0.0 to 1.0).
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.p = max(0.0, min(1.0, p))
        self.rng = random.Random(seed)

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        if candidate_obj >= current_obj:
            return True
        return self.rng.random() < self.p

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"acceptance_probability": self.p}
