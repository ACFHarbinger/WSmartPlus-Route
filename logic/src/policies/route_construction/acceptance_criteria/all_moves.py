"""
All Moves Accepted (AMA) Criterion.
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class AllMovesAccepted(IAcceptanceCriterion):
    """
    Trivial acceptance criterion that accepts every generated neighborhood candidate.
    This effectively transforms the metaheuristic into an unconstrained Random Walk
    across the search space.
    """

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        return True

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
