"""
Fitness Proportional Selection (FPS) Acceptance Criterion.
"""

import random
from typing import Any, Dict, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class FitnessProportionalAcceptance(IAcceptanceCriterion):
    """
    Accepts candidates probabilistically in proportion to their fitness relative
    to the current solution (similar to Roulette-Wheel selection adapted for
    binary choice).

    Probability of accepting the candidate:
    P(accept) = max(0, cand_obj) / (max(0, cur_obj) + max(0, cand_obj))
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        # Prevent negative fitness values
        cur_fit = max(0.0, current_obj)
        cand_fit = max(0.0, candidate_obj)

        total_fit = cur_fit + cand_fit
        if total_fit == 0.0:
            return self.rng.random() < 0.5

        prob_accept = cand_fit / total_fit
        return self.rng.random() < prob_accept

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
