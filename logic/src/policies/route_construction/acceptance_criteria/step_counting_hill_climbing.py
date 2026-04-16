"""
Step Counting Hill Climbing (SCHC) Criterion.
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("schc")
class StepCountingHillClimbing(IAcceptanceCriterion):
    """
    Discrete memory-based criterion.

    The acceptance bound remains fixed for exactly `step_size` iterations, permitting
    local non-improving exploration. Once the limit is reached, the bound explicitly
    updates to match the cost of the *current* incumbent solution.
    """

    def __init__(self, step_size: int):
        """
        Args:
            step_size (int): The number of iterations the bound remains static.
        """
        self.step_size = max(1, step_size)
        self.counter = 0
        self.bound = 0.0

    def setup(self, initial_objective: float) -> None:
        self.bound = initial_objective
        self.counter = 0

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        # Accept if the candidate is better than the frozen bound
        return candidate_obj >= self.bound

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        self.counter += 1
        if self.counter >= self.step_size:
            # Threshold update condition met
            self.bound = current_obj
            self.counter = 0

    def get_state(self) -> Dict[str, Any]:
        return {"bound": self.bound, "steps_until_update": self.step_size - self.counter}
