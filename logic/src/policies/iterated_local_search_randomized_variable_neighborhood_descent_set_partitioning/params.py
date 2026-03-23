"""
ILS-RVND-SP algorithm parameters.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ILSRVNDSPParams:
    """
    Runtime parameters for the ILS-RVND-SP solver.
    Extracts and manages the configuration values during execution.
    """

    max_restarts: int
    max_iter_ils: int
    perturbation_strength: int

    use_set_partitioning: bool
    mip_time_limit: float
    sp_mip_gap: float

    N: int
    A: float
    MaxIter_a: int
    MaxIter_b: int
    MaxIterILS_b: int
    TDev_a: float
    TDev_b: float

    time_limit: float
    seed: Optional[int]
    vrpp: bool = True
    profit_aware_operators: bool = False
    local_search_iterations: int = 500

    @classmethod
    def from_config(cls, config: Any) -> "ILSRVNDSPParams":
        """Build parameters from a configuration object."""
        return cls(
            max_restarts=getattr(config, "max_restarts", 3),
            max_iter_ils=getattr(config, "max_iter_ils", 100),
            perturbation_strength=getattr(config, "perturbation_strength", 2),
            use_set_partitioning=getattr(config, "use_set_partitioning", True),
            mip_time_limit=getattr(config, "mip_time_limit", 30.0),
            sp_mip_gap=getattr(config, "sp_mip_gap", 0.01),
            N=getattr(config, "N", 10),
            A=getattr(config, "A", 0.5),
            MaxIter_a=getattr(config, "MaxIter_a", 10),
            MaxIter_b=getattr(config, "MaxIter_b", 20),
            MaxIterILS_b=getattr(config, "MaxIterILS_b", 50),
            TDev_a=getattr(config, "TDev_a", 0.1),
            TDev_b=getattr(config, "TDev_b", 0.2),
            time_limit=getattr(config, "time_limit", 300.0),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
        )
