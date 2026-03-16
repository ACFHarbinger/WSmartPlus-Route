"""
Threshold Accepting (TA) solver for VRPP.

Accepts deteriorating moves as long as they fall within a specific,
predefined threshold range that decays over time.
"""

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class TASolver(BaseAcceptanceSolver):
    """
    Deterministic non-elitist acceptance criterion.
    Threshold decreases over iterations, eventually becoming Only Improving.
    """

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        # Improving moves always accepted
        if new_profit >= current_profit:
            return True

        progress = iteration / max(self.params.max_iterations - 1, 1)
        threshold = self.params.initial_threshold * (1.0 - progress)

        return new_profit >= current_profit - threshold

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        progress = iteration / max(self.params.max_iterations - 1, 1)
        threshold = self.params.initial_threshold * (1.0 - progress)

        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            threshold=threshold,
        )
