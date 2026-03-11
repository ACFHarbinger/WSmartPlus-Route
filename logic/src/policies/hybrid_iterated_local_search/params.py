"""
HILS algorithm parameters.
"""

from dataclasses import dataclass
from typing import Optional

from logic.src.configs.policies import HILSConfig


@dataclass
class HILSParams:
    """
    Runtime parameters for the HILS solver.
    Extracts and manages the configuration values during execution.
    """

    max_iterations: int
    ils_iterations: int
    perturbation_size: int

    use_set_partitioning: bool
    sp_time_limit: float
    sp_mip_gap: float

    time_limit: float
    seed: Optional[int]

    @classmethod
    def from_config(cls, config: HILSConfig) -> "HILSParams":
        """Build parameters from a HILSConfig instance."""
        seed = getattr(config, "seed", None)

        return cls(
            max_iterations=config.max_iterations,
            ils_iterations=config.ils_iterations,
            perturbation_size=config.perturbation_size,
            use_set_partitioning=config.use_set_partitioning,
            sp_time_limit=config.sp_time_limit,
            sp_mip_gap=config.sp_mip_gap,
            time_limit=config.time_limit,
            seed=seed,
        )
