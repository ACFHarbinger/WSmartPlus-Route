from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


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
        vrpp (bool): Whether the problem is VRP with Profits.
        profit_aware_operators (bool): Whether to use profit-aware insertion/removal.
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
    local_search_iterations: int = 500
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> FILOParams:
        """Create FILOParams from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            FILOParams instance with values from config.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            max_iterations=getattr(config, "max_iterations", 50000),
            initial_temperature_factor=getattr(config, "initial_temperature_factor", 10.0),
            final_temperature_factor=getattr(config, "final_temperature_factor", 100.0),
            shaking_lb_factor=getattr(config, "shaking_lb_factor", 0.5),
            shaking_ub_factor=getattr(config, "shaking_ub_factor", 2.0),
            delta_gamma=getattr(config, "delta_gamma", 0.1),
            gamma_base=getattr(config, "gamma_base", 1.0),
            omega_base_multiplier=getattr(config, "omega_base_multiplier", 1.0),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
