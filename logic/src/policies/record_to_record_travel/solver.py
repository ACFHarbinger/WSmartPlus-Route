"""
Record-to-Record Travel (RRT) for VRPP.

The algorithm tracks the best solution found so far (the "record") and accepts
a candidate solution if its deviation from the record does not exceed a
predefined tolerance threshold. The tolerance decays linearly over iterations,
providing a smooth transition from exploration to exploitation.

Reference:
    Dueck, G., & Scheuer, T. "Threshold Accepting: A General Purpose
    Optimization Algorithm Appearing Superior to Simulated Annealing", 1990.
    Dueck, G. "New Optimization Heuristics: The Great Deluge Algorithm
    and the Record-to-Record Travel", 1993.
"""

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class RRSolver(BaseAcceptanceSolver):
    """
    Record-to-Record Travel solver for VRPP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_tolerance = None

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """
        Accept if within tolerance of the record. Linear decay of tolerance.
        """
        if self.initial_tolerance is None:
            # Tolerance band decays linearly over iterations
            self.initial_tolerance = self.params.tolerance * max(abs(current_profit), 1.0)

        # Linear decay
        progress = iteration / max(self.params.max_iterations - 1, 1)
        self.tolerance = self.initial_tolerance * (1.0 - progress)

        # RR acceptance: accept if within tolerance of the record
        # Note: self.best_profit inherited from BaseAcceptanceSolver
        return new_profit >= self.best_profit - self.tolerance

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            tolerance=getattr(self, "tolerance", 0.0),
        )
