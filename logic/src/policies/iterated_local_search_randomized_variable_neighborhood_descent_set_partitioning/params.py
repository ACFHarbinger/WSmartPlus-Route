"""
ILS-RVND-SP algorithm parameters.
"""

from dataclasses import dataclass
from typing import Optional

from logic.src.configs.policies import ILSRVNDSPConfig


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

    @classmethod
    def from_config(cls, config: ILSRVNDSPConfig) -> "ILSRVNDSPParams":
        """Build parameters from a ILSRVNDSPConfig instance."""
        seed = getattr(config, "seed", None)

        return cls(
            max_restarts=config.max_restarts,
            max_iter_ils=config.max_iter_ils,
            perturbation_strength=config.perturbation_strength,
            use_set_partitioning=config.use_set_partitioning,
            mip_time_limit=config.mip_time_limit,
            sp_mip_gap=config.sp_mip_gap,
            N=config.N,
            A=config.A,
            MaxIter_a=config.MaxIter_a,
            MaxIter_b=config.MaxIter_b,
            MaxIterILS_b=config.MaxIterILS_b,
            TDev_a=config.TDev_a,
            TDev_b=config.TDev_b,
            time_limit=config.time_limit,
            seed=seed,
        )
