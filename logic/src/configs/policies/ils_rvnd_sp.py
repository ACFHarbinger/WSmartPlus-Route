"""
ILS-RVND-SP (Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning) configuration dataclasses.
"""

from dataclasses import dataclass

from .abc import ABCConfig


@dataclass
class ILSRVNDSPConfig(ABCConfig):
    """Configuration for the ILS-RVND-SP algorithm."""

    # ILS Phase parameters
    max_restarts: int = 10
    max_iter_ils: int = 50
    perturbation_strength: int = 2

    # Set Partitioning (SP) Phase parameters
    use_set_partitioning: bool = True
    mip_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    # Paper-specific advanced parameters
    N: int = 150
    A: float = 11.0
    MaxIter_a: int = 50
    MaxIter_b: int = 100
    MaxIterILS_b: int = 2000
    TDev_a: float = 0.05
    TDev_b: float = 0.005

    # Global parameters
    vrpp: bool = True
    time_limit: float = 120.0
    seed: int = 42
    local_search_iterations: int = 500
