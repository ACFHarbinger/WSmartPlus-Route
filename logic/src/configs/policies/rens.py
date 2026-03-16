"""
Relaxation Enforced Neighborhood Search (RENS) configuration.
"""

from dataclasses import dataclass
from typing import Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class RENSConfig:
    """
    Configuration for Relaxation Enforced Neighborhood Search (RENS) matheuristic.

    RENS (Berthold, 2009) is a start heuristic that explores the space of
    feasible roundings of an LP relaxation. It works by fixing all variables
    that take integer values in the LP relaxation and solving a restricted
    MIP on the remaining (fractional) variables.

    Attributes:
        time_limit (float): Total maximum runtime for the heuristic (seconds).
        lp_time_limit (float): Time strictly allocated for the initial LP relaxation phase.
        mip_limit_nodes (int): Node limit for the sub-MIP branch-and-bound tree.
        mip_gap (float): Target optimality gap for the restricted sub-MIP.
        seed (int): Random seed for Gurobi solver consistency.
        engine (str): Infrastructure engine to use (defaults to "custom").
        must_go (Optional[MustGoConfig]): Configuration for mandatory node selection.
        post_processing (Optional[PostProcessingConfig]): Post-optimization cleanup settings.
    """

    time_limit: float = 60.0
    lp_time_limit: float = 10.0
    mip_limit_nodes: int = 1000
    mip_gap: float = 0.01
    seed: int = 42

    # Infrastructure
    engine: str = "custom"
    must_go: Optional[MustGoConfig] = None
    post_processing: Optional[PostProcessingConfig] = None
