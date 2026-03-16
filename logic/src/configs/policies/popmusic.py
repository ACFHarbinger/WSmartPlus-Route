"""
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class POPMUSICConfig:
    """Configuration for POPMUSIC matheuristic.
    Based on Taillard & Voss (2002).

    Attributes:
        subproblem_size: Number of neighboring routes to optimize (R).
        max_iterations: Maximum number of sub-problem attempts without improvement.
        base_solver: The solver configuration used for subproblem optimization.
        initial_solver: Solver used to generate the initial solution (e.g., 'nearest_neighbor').
        seed: Random seed for reproducibility.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    # Core POPMUSIC parameters
    subproblem_size: int = 3
    max_iterations: int = 100

    # Solver orchestration
    base_solver: str = "alns"
    initial_solver: str = "nearest_neighbor"
    seed: Optional[int] = None

    # Infrastructure
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
