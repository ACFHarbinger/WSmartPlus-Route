"""
Branch-and-Bound (BB) configuration schema.

This module defines the dataclass used to configure the Branch-and-Bound policy,
mapping YAML configuration values to typed Python attributes for use in the solver.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class BBConfig:
    """Configuration for the Branch-and-Bound (BB) solver.

    The BB solver is an exact algorithm that explores a state-space search tree
    using Linear Programming (LP) relaxations to obtain lower bounds and prune
    branches that cannot yield a solution better than the current incumbent.

    Attributes:
        time_limit (float): Maximum allowed runtime for the solver in seconds.
            Defaults to 60.0.
        mip_gap (float): Relative tolerance for the optimality gap. The solver
            terminates if (Incumbent - Bound) / Incumbent < mip_gap.
            Defaults to 0.01.
        branching_strategy (str): Logic for selecting the fractional variable to
            branch on. Supported: 'most_fractional', 'least_fractional'.
            Defaults to "most_fractional".
        vrpp (bool): If True, enables the Vehicle Routing Problem with Profits
            mode, where all bins are candidates for collection based on profitability.
            Defaults to True.
        seed (Optional[int]): Random seed passed to the underlying Gurobi engine
            for deterministic results. Defaults to None.
        must_go (Optional[List[MustGoConfig]]): List of strategies determining
            mandatory nodes to visit. Defaults to None.
        post_processing (Optional[List[PostProcessingConfig]]): List of operations
            to refine the tour after the exact search completes. Defaults to None.
    """

    time_limit: float = 60.0
    mip_gap: float = 0.01
    branching_strategy: str = "most_fractional"
    vrpp: bool = True
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
