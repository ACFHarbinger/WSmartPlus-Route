"""
Late Acceptance Hill Climbing (LAHC) Criterion.
"""

from typing import Any, Dict, List

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("lahc")
class LateAcceptance(IAcceptanceCriterion):
    """
    Memory-based thresholding utilizing a finite circular array.

    A candidate is accepted if its objective is strictly improving against the
    current solution OR if it is better than the cost encountered exactly L steps ago.
    Reference: Burke & Bykov (2017).
    """

    def __init__(self, queue_size: int):
        """
        Args:
            queue_size (int): The length of the historical memory queue (L).
        """
        self.L = max(1, queue_size)
        self.history: List[float] = []
        self.pointer = 0

    def setup(self, initial_objective: float) -> None:
        # Populate the entire circular array with the starting objective
        self.history = [initial_objective] * self.L
        self.pointer = 0

    def accept(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        # Accept if improving vs current OR improving vs L steps ago
        return candidate_obj >= current_obj or candidate_obj >= self.history[self.pointer]

    def step(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        # Always insert the *current accepted state's* cost into the array
        self.history[self.pointer] = current_obj
        self.pointer = (self.pointer + 1) % self.L

    def get_state(self) -> Dict[str, Any]:
        return {"pointer": self.pointer, "current_history_val": self.history[self.pointer]}
