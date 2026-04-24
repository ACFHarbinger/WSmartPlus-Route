"""Aspiration Criterion (AC).

Accepts candidates that surpass the best objective found so far, regardless
of other criteria.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ac")
class AspirationCriterion(IAcceptanceCriterion):
    """Memory-overriding acceptance criterion.

    Accepts a candidate only if its objective surpasses the global best objective
    found since the criterion was initialized.

    Attributes:
        global_best (float): The best objective value observed during the search.
    """

    def __init__(self) -> None:
        """Initialize the Aspiration criterion."""
        self.global_best = -float("inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the global best with the starting objective.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.global_best = initial_objective

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Accept if the candidate surpasses the global best objective.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            **kwargs (Any): Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate_obj > global_best.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj > self.global_best)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "global_best_objective": self.global_best,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Update the global best objective if a better solution was accepted.

        Args:
            current_obj (ObjectiveValue): Previous solution's objective.
            candidate_obj (ObjectiveValue): Candidate solution's objective.
            accepted (bool): Whether the candidate was accepted.
            **kwargs (Any): Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted and candidate_obj > self.global_best:
            self.global_best = candidate_obj

    def get_state(self) -> Dict[str, Any]:
        """Return the current global best objective.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"global_best_objective": self.global_best}
