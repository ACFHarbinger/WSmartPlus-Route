"""
HILS (Hybrid Iterated Local Search) configuration dataclasses.
"""

from dataclasses import dataclass

from .abc import ABCConfig


@dataclass
class HILSConfig(ABCConfig):
    """Configuration for the Hybrid Iterated Local Search algorithm."""

    # ILS Phase parameters
    max_iterations: int = 100
    ils_iterations: int = 50
    perturbation_size: int = 2

    # Set Partitioning (SP) Phase parameters
    use_set_partitioning: bool = True
    sp_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    # Global parameters
    time_limit: float = 120.0
    seed: int = 42
