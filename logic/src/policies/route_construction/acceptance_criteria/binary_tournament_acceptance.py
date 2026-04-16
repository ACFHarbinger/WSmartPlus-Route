"""
Simulated Tournament Acceptance (STA) Criterion.
"""

import random
from typing import Any, Dict, Optional

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("bta")
class BinaryTournamentAcceptance(IAcceptanceCriterion):
    """
    Binary Tournament stochastic acceptance.
    Evaluates the current solution against the candidate solution. The superior
    solution "wins" the tournament with probability `p` (where `p` > 0.5), and
    the inferior solution wins with probability `1 - p`.

    If the candidate wins, it is accepted.
    """

    def __init__(self, p: float = 0.8, seed: Optional[int] = None) -> None:
        """
        Args:
            p (float): The probability that the superior solution wins the tournament.
                Typically between 0.5 and 1.0.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.p = p
        self.rng = random.Random(seed)

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        roll = self.rng.random()
        candidate_is_superior = candidate_obj > current_obj

        if candidate_is_superior:
            # Candidate gets the high probability
            return roll < self.p
        else:
            # Candidate is worse, so it gets the low probability
            return roll > self.p

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"tournament_probability": self.p}
