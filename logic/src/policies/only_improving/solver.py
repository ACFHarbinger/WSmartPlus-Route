"""
Only Improving (OI) solver for VRPP.

Accepts a candidate move if and only if it strictly improves the current solution.
"""

from ..other.local_search.base_acceptance_criteria import BaseAcceptanceSolver


class OISolver(BaseAcceptanceSolver):
    """
    Strictest elitist acceptance criterion.
    """

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        return new_profit > current_profit
