"""
Boltzmann-Metropolis Criterion (BMC) for VRPP.

Classic meta-heuristic drawing analogy from metallurgical annealing.
Non-improving moves are accepted with Boltzmann probability
exp(Δprofit / T), where T is a temperature parameter that decays
geometrically via T *= alpha.

Reference:
    Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P.
    "Optimization by Simulated Annealing", 1983.
"""

import math

from ..base.base_acceptance_criteria import BaseAcceptanceSolver


class BMCSolver(BaseAcceptanceSolver):
    """
    Boltzmann-Metropolis Criterion solver for VRPP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = self.params.initial_temp

    def _accept(self, new_profit: float, current_profit: float, iteration: int) -> bool:
        """
        Stochastic acceptance via Boltzmann distribution. Geometric cooling.
        """
        if new_profit >= current_profit:
            return True

        delta = new_profit - current_profit
        # Probability of acceptance: exp(delta / T)
        # Note: delta is negative here.
        prob = math.exp(delta / self.T) if self.T > 0 else 0.0

        return self.random.random() < prob

    def _update_state(self, iteration: int):
        # Geometric cooling
        self.T *= self.params.alpha
        if self.params.min_temp > self.T:
            self.T = self.params.min_temp

    def _record_telemetry(self, iteration: int, best_profit: float, current_profit: float):
        self._viz_record(
            iteration=iteration,
            best_profit=best_profit,
            current_profit=current_profit,
            temp=self.T,
        )
