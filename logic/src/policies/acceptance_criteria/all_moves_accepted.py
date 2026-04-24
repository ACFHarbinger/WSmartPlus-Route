"""All Moves Accepted (AMA) Criterion.

Trivial criterion used for random walks or testing.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ama")
class AllMovesAccepted(IAcceptanceCriterion):
    """Trivial acceptance criterion that accepts every generated candidate.

    This effectively transforms the metaheuristic into an unconstrained
    Random Walk across the search space.

    Attributes:
        None
    """

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Unconditionally accept the move.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            **kwargs (Any): Additional context (not used).

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): Always True.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(True)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """No-op update step.

        Args:
            current_obj (ObjectiveValue): Objective of the previous solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            accepted (bool): Whether the candidate was accepted.
            **kwargs (Any): Additional context (not used).
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        """Return an empty state dictionary.

        Returns:
            Dict[str, Any]: Empty dictionary.
        """
        return {}
