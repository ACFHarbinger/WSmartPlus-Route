"""
Improving and Equal (IE) Criterion.
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ie")
class ImprovingAndEqual(IAcceptanceCriterion):
    """
    Weakly elitist strategy.

    Accepts improving moves and identical-cost moves, allowing the solver to
    transverse neutral objective plateaus.
    """

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        return candidate_obj >= current_obj

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
