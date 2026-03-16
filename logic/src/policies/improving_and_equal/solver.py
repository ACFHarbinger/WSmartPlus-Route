"""
Improving and Equal (IE) solver for VRPP.

Accepts candidate moves if their profit is greater than or equal to the current profit.
Allows traversing plateau regions in the search space.
"""

from ..other.local_search.base_acceptance_criteria import BaseAcceptanceSolver


class IESolver(BaseAcceptanceSolver):
    """
    Elitist acceptance criterion that allows equal moves.
    """

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        return new_profit >= current_profit
