"""
Configuration for the Local Branching (LB) matheuristic.
"""

from dataclasses import dataclass
from typing import Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class LocalBranchingConfig:
    """
    Configuration for the Local Branching (LB) matheuristic.

    Local Branching (Fischetti and Lodi, 2003) is a matheuristic that uses a
    general-purpose MIP solver to explore the k-neighborhood of a given
    incumbent solution.

    Attributes:
        time_limit (float): Total wall-clock time limit for the policy.
        time_limit_per_iteration (float): Time limit for each sub-MIP solve.
        k (int): Neighborhood size (Hamming distance).
        max_iterations (int): Maximum number of improvement iterations.
        node_limit_per_iteration (int): Branch-and-bound node limit per sub-MIP.
        mip_gap (float): Targeted relative optimality gap for sub-problems.
        seed (int): Random seed for reproducibility.
        vrpp (bool): Whether the problem is a VRP with Profits.

        # Infrastructure
        engine (str): Identifier for the optimization engine (default: "custom").
        must_go (Optional[MustGoConfig]): Configuration for mandatory node selection.
        post_processing (Optional[PostProcessingConfig]): Optional local search refinement.
    """

    time_limit: float = 300.0
    time_limit_per_iteration: float = 30.0
    k: int = 10
    max_iterations: int = 20
    node_limit_per_iteration: int = 5000
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Infrastructure
    engine: str = "custom"
    must_go: Optional[MustGoConfig] = None
    post_processing: Optional[PostProcessingConfig] = None
