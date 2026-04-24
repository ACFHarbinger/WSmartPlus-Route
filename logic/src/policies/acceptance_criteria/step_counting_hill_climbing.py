"""Step Counting Hill Climbing (SCHC) Criterion.

Discrete memory-based criterion that updates the acceptance bound periodically.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("schc")
class StepCountingHillClimbing(IAcceptanceCriterion):
    """Discrete memory-based criterion.

    The acceptance bound remains fixed for exactly `step_size` iterations, permitting
    local non-improving exploration. Once the limit is reached, the bound explicitly
    updates to match the cost of the *current* incumbent solution.

    Attributes:
        step_size (int): The number of iterations the bound remains static.
        counter (int): Current iteration counter.
        bound (float): Current acceptance threshold.
    """

    def __init__(self, step_size: int):
        """Initialize the SCHC criterion.

        Args:
            step_size (int): The number of iterations the bound remains static.
        """
        self.step_size = max(1, step_size)
        self.counter = 0
        self.bound = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        """Initialize the criterion state.

        Args:
            initial_objective (ObjectiveValue): The initial solution's objective.
        """
        initial_objective = cast(float, initial_objective)
        self.bound = initial_objective
        self.counter = 0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        """Determine whether to accept based on the current frozen bound.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            **kwargs (Any): Additional context (not used).

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate improves the bound.
                - metrics (AcceptanceMetrics): State metadata including current bound.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Accept if the candidate is better than the frozen bound
        _accepted = bool(candidate_obj >= self.bound)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "bound": self.bound,
            "steps_until_update": self.step_size - self.counter,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Increment the counter and update the bound if step_size is reached.

        Args:
            current_obj (ObjectiveValue): Objective of the previous solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            accepted (bool): Whether the candidate was accepted.
            **kwargs (Any): Additional context (not used).
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.counter += 1
        if self.counter >= self.step_size:
            # Threshold update condition met
            self.bound = current_obj
            self.counter = 0

    def get_state(self) -> Dict[str, Any]:
        """Return the current bound and steps remaining.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"bound": self.bound, "steps_until_update": self.step_size - self.counter}
