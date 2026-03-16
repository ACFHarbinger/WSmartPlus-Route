"""
Great Deluge (GD) solver for VRPP.

Accepts a candidate move if its objective value is better than a
monotonically updating threshold known as the "water level".
"""

import time

from ..other.local_search.base_acceptance_criteria import BaseAcceptanceSolver


class GDSolver(BaseAcceptanceSolver):
    """
    Deterministic non-elitist acceptance criterion.
    Water level rises over time, forcing the solver to find better solutions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f0 = None
        self.target_f = None
        self.wall_start = time.process_time()

    def _update_state(self, iteration: int):
        if self.f0 is None:
            # We need the initial profit after build_initial_solution is called in solve()
            # Since solve() calls _update_state before LLH, we can catch it here.
            # However, self.routes is not yet populated by the base class at the very first step.
            # We'll initialize in the first _accept call if needed or wrap build_initial_solution.
            pass

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        if self.f0 is None:
            self.f0 = current_profit
            self.target_f = self.f0 * self.params.target_fitness_multiplier
            self.wall_start = time.process_time()

        # Improving moves are always accepted
        if new_profit > current_profit:
            return True

        # Great Deluge logic: accept if above water level
        elapsed = time.process_time() - self.wall_start
        progress = (
            min(1.0, elapsed / self.params.time_limit)
            if self.params.time_limit > 0
            else (iteration / self.params.max_iterations)
        )

        water_level = self.f0 + (self.target_f - self.f0) * progress

        return new_profit >= water_level

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        elapsed = time.process_time() - self.wall_start
        progress = (
            min(1.0, elapsed / self.params.time_limit)
            if self.params.time_limit > 0
            else (iteration / self.params.max_iterations)
        )
        water_level = (self.f0 + (self.target_f - self.f0) * progress) if self.f0 is not None else 0.0

        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            water_level=water_level,
        )
