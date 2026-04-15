"""
Only Improving (OI) Criterion.
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class OnlyImproving(IAcceptanceCriterion):
    """
    Strictest form of greedy, elitist move acceptance.
    Accepts a candidate only if it yields a strict objective enhancement.
    """

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float) -> bool:
        return candidate_obj > current_obj

    def step(self, current_obj: float, candidate_obj: float, accepted: bool) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
