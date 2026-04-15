"""
Ensemble Move Acceptance (EMA) Criterion.
"""

from typing import Any, Dict, List

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


class EnsembleAcceptance(IAcceptanceCriterion):
    """
    Meta-decision architecture combining multiple criteria.

    Evaluates a candidate solution through a portfolio of heterogeneous criteria
    and aggregates the decision via logical ensemble rules (e.g., G-AND, G-OR, G-VOT).
    """

    def __init__(self, criteria: List[IAcceptanceCriterion], rule: str = "G-VOT"):
        """
        Args:
            criteria (List[IAcceptanceCriterion]): The instantiated sub-criteria.
            rule (str): The aggregation logic ('G-AND', 'G-OR', 'G-VOT').
        """
        if not criteria:
            raise ValueError("EnsembleAcceptance requires at least one initialized criterion.")
        self.criteria = criteria
        self.rule = rule.upper()

    def setup(self, initial_objective: float) -> None:
        for crit in self.criteria:
            crit.setup(initial_objective)

    def accept(self, current_obj: float, candidate_obj: float) -> bool:
        # Gather votes from all encapsulated criteria
        votes = [crit.accept(current_obj, candidate_obj) for crit in self.criteria]

        if self.rule == "G-AND":
            return all(votes)  # Strict consensus
        elif self.rule == "G-OR":
            return any(votes)  # Authority rule (at least one)
        elif self.rule == "G-VOT":
            return sum(votes) >= (len(votes) / 2.0)  # Majority rule
        else:
            raise ValueError(f"Unknown EMA rule: {self.rule}")

    def step(self, current_obj: float, candidate_obj: float, accepted: bool) -> None:
        # Advance the state of all encapsulated criteria
        for crit in self.criteria:
            crit.step(current_obj, candidate_obj, accepted)

    def get_state(self) -> Dict[str, Any]:
        # Aggregate the states of the underlying criteria
        return {f"criterion_{i}": crit.get_state() for i, crit in enumerate(self.criteria)}
