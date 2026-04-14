"""
Progressive Hedging (PH) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class PHConfig:
    """Configuration for Progressive Hedging (PH) policy.

    Progressive Hedging (Rockafellar and Wets, 1991) is a horizontal
    decomposition algorithm for stochastic programming. It decomposes the
    stochastic VRP into scenario-specific subproblems and iteratively enforces
    non-anticipativity constraints.

    Attributes:
        rho: Penalty parameter for the quadratic non-anticipativity term.
            Larger values enforce consensus faster but may lead to suboptimality.
        max_iterations: Maximum number of PH iterations.
        convergence_tol: Tolerance for consensual visit decisions (non-anticipativity error).
        sub_solver: Key of the deterministic solver to use for subproblems (e.g., 'bc', 'alns').
        num_scenarios: Number of scenarios for SAA if none are provided.
        time_limit: Total wall-clock time limit in seconds.
        verbose: Enable detailed logging of PH iterations and convergence stats.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
        seed: Random seed for reproducibility.
    """

    rho: float = 1.0
    max_iterations: int = 50
    convergence_tol: float = 0.01
    sub_solver: str = "bc"
    num_scenarios: int = 10
    time_limit: float = 300.0
    verbose: bool = True
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
    seed: Optional[int] = None
