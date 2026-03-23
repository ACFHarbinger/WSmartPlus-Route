"""
Configuration parameters for the Simulated Annealing (SA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SAParams:
    """
    Configuration for the SA solver.

    Classic Boltzmann acceptance with geometric cooling schedule.

    Attributes:
        initial_temp: Starting temperature.
        alpha: Geometric cooling factor (T *= alpha each iteration).
        min_temp: Temperature floor below which cooling stops.
        max_iterations: Total LLH applications.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRPP (True) or CVRP (False).
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    initial_temp: float = 100.0
    alpha: float = 0.995
    min_temp: float = 0.01
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> SAParams:
        """Build parameters from a configuration object."""
        return cls(
            initial_temp=getattr(config, "initial_temp", 100.0),
            alpha=getattr(config, "alpha", 0.995),
            min_temp=getattr(config, "min_temp", 0.01),
            max_iterations=getattr(config, "max_iterations", 500),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
