"""
Threshold Accepting (TA) parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TAParams:
    """
    Configuration for the Threshold Accepting (TA) solver.

    Attributes:
        max_iterations: Total LLH applications.
        initial_threshold: Starting absolute tolerance.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per destroy step.
        n_llh: Number of LLHs in the pool.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRPP (True) or CVRP (False).
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    max_iterations: int = 1000
    initial_threshold: float = 100.0
    time_limit: float = 60.0
    n_removal: int = 2
    n_llh: int = 5
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> TAParams:
        """Build parameters from a configuration object."""
        return cls(
            max_iterations=getattr(config, "max_iterations", 1000),
            initial_threshold=getattr(config, "initial_threshold", 100.0),
            time_limit=getattr(config, "time_limit", 60.0),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
