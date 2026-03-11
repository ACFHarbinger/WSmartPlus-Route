"""
FILO (Fast Iterative Localized Optimization) configuration dataclasses.
"""

from dataclasses import dataclass

from .abc import ABCConfig


@dataclass
class FILOConfig(ABCConfig):
    """
    Configuration parameters for the Fast Iterative Localized Optimization (FILO) solver.

    Attributes:
        time_limit: Maximum allowed runtime in seconds.
        max_iterations: Maximum number of core optimization iterations.
        initial_temperature_factor: Divisor for initial temperature calculation (e.g. 10.0 -> mean_cost / 10).
        final_temperature_factor: Divisor for final temperature calculation.
        shaking_lb_factor: Lower bound intensification factor for the shaking phase.
        shaking_ub_factor: Upper bound intensification factor for the shaking phase.
        delta_gamma: Update factor for gamma node activation penalties.
        gamma_base: Base probability for neighborhood evaluation.
        omega_base_multiplier: Base shaking intensity multiplier.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 60.0
    max_iterations: int = 50000
    initial_temperature_factor: float = 10.0
    final_temperature_factor: float = 100.0
    shaking_lb_factor: float = 0.5
    shaking_ub_factor: float = 2.0
    delta_gamma: float = 0.1
    gamma_base: float = 1.0
    omega_base_multiplier: float = 1.0
    seed: int = 42
