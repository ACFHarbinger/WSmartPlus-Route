"""
Step Counting Hill Climbing (SCHC) solver for VRPP.

Accepts a candidate move if its objective value is better than a
threshold that stays fixed for a specific number of steps before
updating to the current solution's cost.
"""

from ..other.local_search.base_acceptance_criteria import BaseAcceptanceSolver


class SCHCSolver(BaseAcceptanceSolver):
    """
    Memory-based acceptance criterion.
    Threshold remains constant for 'step_size' iterations, then resets to current profit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = None

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """Accept if better than or equal to threshold; update threshold every L steps."""
        if self.threshold is None:
            self.threshold = current_profit

        if iteration > 0 and iteration % self.params.step_size == 0:
            self.threshold = current_profit

        return new_profit >= self.threshold

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            threshold=self.threshold if self.threshold is not None else 0.0,
        )
