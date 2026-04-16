"""
Aspiration Criterion (AC).
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ac")
class AspirationCriterion(IAcceptanceCriterion):
    """
    Strictest form of memory-overriding acceptance.
    Accepts a candidate only if its objective surpasses the global best objective
    found since the criterion was initialized.
    """

    def __init__(self) -> None:
        self.global_best = -float("inf")

    def setup(self, initial_objective: float) -> None:
        self.global_best = initial_objective

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        return candidate_obj > self.global_best

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        if accepted and candidate_obj > self.global_best:
            self.global_best = candidate_obj

    def get_state(self) -> Dict[str, Any]:
        return {"global_best_objective": self.global_best}
