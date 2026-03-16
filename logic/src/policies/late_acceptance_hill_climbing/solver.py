"""
Late Acceptance Hill-Climbing (LAHC) for VRPP.

Instead of comparing a candidate solution against the current solution,
LAHC compares it against the solution from ``L`` iterations ago, stored
in a circular queue. This deferred comparison induces a dynamic cooling
effect without requiring explicit temperature scheduling.

Reference:
    Burke, E. K., & Bykov, Y. "The Late Acceptance Hill-Climbing Heuristic", 2016.
"""

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class LAHCSolver(BaseAcceptanceSolver):
    """
    Late Acceptance Hill-Climbing solver for VRPP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Circular queue initialised to the starting profit
        self.L = self.params.queue_size
        self.queue = [0.0] * self.L
        self.initialized = False

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """
        Accept if better than current OR better than the solution from L iterations ago.
        """
        if not self.initialized:
            self.queue = [current_profit] * self.L
            self.initialized = True

        v = iteration % self.L
        prev_f = self.queue[v]

        if new_profit >= current_profit or new_profit >= prev_f:
            # Update circular queue with the profit of the solution that is now accepted
            # Note: The original implementation updates queue[v] with profit AFTER acceptance
            # but before the next iteration.
            self.queue[v] = new_profit
            return True

        # If rejected, we update the queue with the CURRENT profit (which stayed the same)
        self.queue[v] = current_profit
        return False

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        v = iteration % self.L
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            queue_entry=self.queue[v],
        )
