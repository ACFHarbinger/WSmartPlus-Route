"""Threshold Accepting (TA) Criterion.

Deterministic counterpart to Simulated Annealing that explicitly bounds
allowable deterioration via a decaying threshold.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ta")
class ThresholdAccepting(IAcceptanceCriterion):
    """Deterministic Threshold Accepting.

    A deterministic counterpart to Simulated Annealing. It explicitly bounds the
    allowable deterioration using a threshold parameter that decays linearly
    towards zero over the specified iteration budget.

    Attributes:
        threshold (float): Current absolute tolerance for deterioration.
        decay_rate (float): Amount by which the threshold is reduced at each step.
    """

    def __init__(self, initial_threshold: float, max_iterations: int):
        """Initialize the Threshold Accepting criterion.

        Args:
            initial_threshold (float): The starting absolute tolerance for
                deterioration.
            max_iterations (int): The budget used to calculate linear decay.
        """
        self.threshold = initial_threshold
        self.decay_rate = initial_threshold / max_iterations if max_iterations > 0 else 0.0

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
        """Determine acceptance based on the current decaying threshold.

        Args:
            current_obj (ObjectiveValue): Objective of the current solution.
            candidate_obj (ObjectiveValue): Objective of the candidate solution.
            **kwargs (Any): Additional context.

        Returns:
            Tuple[bool, AcceptanceMetrics]: A tuple containing:
                - accepted (bool): True if candidate is within threshold tolerance.
                - metrics (AcceptanceMetrics): Performance metadata.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # delta = candidate - current. If worsening by 10, delta is -10.
        # If threshold is 15, -10 >= -15 is True.
        delta = candidate_obj - current_obj
        _accepted = bool(delta >= -self.threshold)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "threshold": self.threshold}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        """Apply linear decay to the threshold.

        Args:
            current_obj (ObjectiveValue): Previous solution's objective.
            candidate_obj (ObjectiveValue): Candidate solution's objective.
            accepted (bool): Whether the candidate was accepted.
            **kwargs (Any): Additional context.
        """
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.threshold = max(0.0, self.threshold - self.decay_rate)

    def get_state(self) -> Dict[str, Any]:
        """Return the current threshold.

        Returns:
            Dict[str, Any]: State dictionary.
        """
        return {"threshold": self.threshold}
