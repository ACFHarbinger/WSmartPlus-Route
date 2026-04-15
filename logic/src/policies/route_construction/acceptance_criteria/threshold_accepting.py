"""
Threshold Accepting (TA) Criterion.
"""

from typing import Any, Dict

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class ThresholdAccepting(IAcceptanceCriterion):
    """
    Deterministic Threshold Accepting.

    A deterministic counterpart to Simulated Annealing. It explicitly bounds the
    allowable deterioration using a threshold parameter that decays linearly
    towards zero over the specified iteration budget.
    """

    def __init__(self, initial_threshold: float, max_iterations: int):
        """
        Args:
            initial_threshold (float): The starting absolute tolerance for deterioration.
            max_iterations (int): The budget used to calculate linear decay.
        """
        self.threshold = initial_threshold
        self.decay_rate = initial_threshold / max_iterations if max_iterations > 0 else 0.0

    def setup(self, initial_objective: float) -> None:
        pass

    def accept(self, current_obj: float, candidate_obj: float) -> bool:
        # delta = candidate - current. If worsening by 10, delta is -10.
        # If threshold is 15, -10 >= -15 is True.
        delta = candidate_obj - current_obj
        return delta >= -self.threshold

    def step(self, current_obj: float, candidate_obj: float, accepted: bool) -> None:
        self.threshold = max(0.0, self.threshold - self.decay_rate)

    def get_state(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}
