"""
BC (Branch-and-Cut) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class BCConfig:
    """Configuration for Branch-and-Cut (BC) policy.

    Branch-and-Cut is an exact optimization method that combines:
    - Cutting planes to strengthen LP relaxations
    - Branch-and-bound for integer optimization
    - Gurobi solver for MIP solving

    The algorithm solves the VRPP to provable optimality using:
    1. Initial LP relaxation
    2. Separation algorithms for violated inequalities:
       - Subtour elimination constraints (SEC)
       - Capacity inequalities
       - Comb inequalities (optional)
    3. Branch-and-bound when cuts are insufficient

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        mip_gap: Relative MIP optimality gap (0.0 = prove optimality).
        use_heuristics: Whether to use primal heuristics for warm start.
        use_exact_separation: Use exact max-flow for SEC (slower but stronger).
        max_cuts_per_round: Maximum cuts to add per separation round.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 300.0
    mip_gap: float = 0.0
    use_heuristics: bool = True
    use_exact_separation: bool = False
    max_cuts_per_round: int = 50
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
    seed: Optional[int] = None
