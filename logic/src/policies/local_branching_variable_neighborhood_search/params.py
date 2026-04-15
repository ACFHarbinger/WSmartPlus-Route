"""
Configuration parameters for the Local Branching Variable Neighborhood Search (LB-VNS).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class LBVNSParams:
    """
    Configuration parameters for LB-VNS.

    Attributes:
        k_min: Minimum neighborhood size.
        k_max: Maximum neighborhood size.
        k_step: Step size for neighborhood expansion.
        time_limit: Total time budget for optimization.
        time_limit_per_lb: Time limit for each Local Branching solve.
        max_lb_iterations: Maximum number of LB iterations per VNS iteration.
        mip_gap: Acceptable relative optimality gap.
        seed: Random seed.
    """

    k_min: int = 10
    k_max: int = 30
    k_step: int = 5
    time_limit: float = 300.0
    time_limit_per_lb: float = 60.0
    max_lb_iterations: int = 5
    mip_gap: float = 0.01
    seed: int = 42
    engine: str = "gurobi"
    framework: str = "ortools"

    @classmethod
    def from_config(cls, config: Any) -> LBVNSParams:
        """Create LBVNSParams from a configuration object or dictionary."""
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            k_min=getattr(config, "k_min", 10),
            k_max=getattr(config, "k_max", 30),
            k_step=getattr(config, "k_step", 5),
            time_limit=getattr(config, "time_limit", 300.0),
            time_limit_per_lb=getattr(config, "time_limit_per_lb", 60.0),
            max_lb_iterations=getattr(config, "max_lb_iterations", 5),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert LBVNSParams to a dictionary for backend compatibility."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
