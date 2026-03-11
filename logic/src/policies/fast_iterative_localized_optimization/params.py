"""
Configuration parameters for Fast Iterative Localized Optimization (FILO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies.filo import FILOConfig


@dataclass
class FILOParams:
    """
    Configuration parameters for the FILO solver context.

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

    @classmethod
    def from_config(cls, config: FILOConfig) -> FILOParams:
        """Create FILOParams from a FILOConfig dataclass.

        Args:
            config: FILOConfig dataclass with solver parameters.

        Returns:
            FILOParams instance with values from config.
        """
        return cls(
            time_limit=config.time_limit,
            max_iterations=config.max_iterations,
            initial_temperature_factor=config.initial_temperature_factor,
            final_temperature_factor=config.final_temperature_factor,
            shaking_lb_factor=config.shaking_lb_factor,
            shaking_ub_factor=config.shaking_ub_factor,
            delta_gamma=config.delta_gamma,
            gamma_base=config.gamma_base,
            omega_base_multiplier=config.omega_base_multiplier,
            seed=config.seed,
        )
