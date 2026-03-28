"""
BPC (Branch-and-Price-and-Cut) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class BPCConfig:
    """Configuration for Branch-and-Price-and-Cut (BPC) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('ortools', 'gurobi', 'custom').
        profit_aware_operators: Whether to use profit-aware operators.
        vrpp: Whether to use VRPP.
        seed: Random seed for reproducibility.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
        search_strategy: B&B node selection strategy ('best_first' or 'depth_first').
        cutting_planes: Cutting plane family ('rcc' or 'lci').
        branching_strategy: Branching rule ('ryan_foster', 'edge', or 'divergence').
    """

    time_limit: float = 60.0
    engine: str = "ortools"
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
    search_strategy: str = "best_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "ryan_foster"
