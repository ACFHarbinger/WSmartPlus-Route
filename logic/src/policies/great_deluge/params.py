"""
Great Deluge (GD) parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GDParams:
    """
    Configuration for the Great Deluge (GD) solver.

    Attributes:
        max_iterations: Total LLH applications.
        target_fitness_multiplier: Initial target = initial_profit * multiplier.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRPP (True) or CVRP (False).
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    max_iterations: int = 1000
    target_fitness_multiplier: float = 1.1
    time_limit: float = 60.0
    n_removal: int = 2
    n_llh: int = 5
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> GDParams:
        """Build parameters from a configuration object."""
        return cls(
            max_iterations=getattr(config, "max_iterations", 1000),
            target_fitness_multiplier=getattr(config, "target_fitness_multiplier", 1.1),
            time_limit=getattr(config, "time_limit", 60.0),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
