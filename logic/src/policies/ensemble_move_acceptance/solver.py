"""
Ensemble Move Acceptance (EMA) solver for VRPP.

Combines multiple move acceptance criteria (e.g., SA, GD, IE)
to make a joint acceptance decision using rules like G-AND, G-OR,
G-VOT, or G-PVO.
"""

import math
import time

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class EMASolver(BaseAcceptanceSolver):
    """
    Group decision-making approach for move acceptance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize sub-state
        self.wall_start = time.process_time()
        self.f0 = None

        # SA state
        self.temp = self.params.sub_params.get("sa", {}).get("initial_temp", 100.0)

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        if self.f0 is None:
            self.f0 = current_profit
            self.wall_start = time.process_time()

        decisions = []
        for crit in self.params.criteria:
            if crit == "oi":
                decisions.append(new_profit > current_profit)
            elif crit == "ie":
                decisions.append(new_profit >= current_profit)
            elif crit == "sa":
                decisions.append(self._check_sa(new_profit, current_profit))
            elif crit == "gd":
                decisions.append(self._check_gd(new_profit, current_profit, iteration))
            elif crit == "ta":
                decisions.append(self._check_ta(new_profit, current_profit, iteration))

        if not decisions:
            return new_profit > current_profit

        rule = self.params.rule
        if rule == "G-AND":
            return all(decisions)
        elif rule == "G-OR":
            return any(decisions)
        elif rule == "G-VOT":
            return sum(decisions) > len(decisions) / 2
        elif rule == "G-PVO":
            prob = sum(decisions) / len(decisions)
            return self.random.random() < prob

        return any(decisions)

    def _update_state(self, iteration: int):
        # Update SA temp
        sa_p = self.params.sub_params.get("sa", {})
        alpha = sa_p.get("alpha", 0.995)
        min_temp = sa_p.get("min_temp", 0.01)
        self.temp = max(min_temp, self.temp * alpha)

    # --- Sub-criteria checks ---

    def _check_sa(self, new_profit, current_profit) -> bool:
        delta = new_profit - current_profit
        if delta >= 0:
            return True
        if self.temp > 1e-10:
            return self.random.random() < math.exp(delta / self.temp)
        return False

    def _check_gd(self, new_profit, current_profit, iteration) -> bool:
        if self.f0 is None:
            return True
        gd_p = self.params.sub_params.get("gd", {})
        target_f = self.f0 * gd_p.get("target_fitness_multiplier", 1.1)

        elapsed = time.process_time() - self.wall_start
        progress = (
            min(1.0, elapsed / self.params.time_limit)
            if self.params.time_limit > 0
            else (iteration / self.params.max_iterations)
        )
        water_level = self.f0 + (target_f - self.f0) * progress
        return new_profit >= water_level

    def _check_ta(self, new_profit, current_profit, iteration) -> bool:
        if new_profit >= current_profit:
            return True
        ta_p = self.params.sub_params.get("ta", {})
        initial_threshold = ta_p.get("initial_threshold", 100.0)
        progress = iteration / max(self.params.max_iterations - 1, 1)
        threshold = initial_threshold * (1.0 - progress)
        return new_profit >= current_profit - threshold

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            ensemble_rule=self.params.rule,
        )
