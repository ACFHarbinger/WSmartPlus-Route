"""
GIHH (Hyper-Heuristic with Two Guidance Indicators) configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class GIHHConfig:
    """Configuration for GIHH (Hyper-Heuristic with Two Guidance Indicators) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
        max_iterations: Maximum number of iterations.
        vrpp: If True, enable VRPP mode (visit subset profitably).

        # Operators
        move_operators: List of move operator names for local search.
        perturbation_operators: List of perturbation operator names.

        # Guidance Indicator weights
        iri_weight: Weight for Improvement Rate Indicator (0.0-1.0).
        tbi_weight: Weight for Time-based Indicator (0.0-1.0).

        # Learning parameters
        learning_rate: Rate of indicator updates (0.0-1.0).
        memory_size: Number of recent iterations to track.
        epsilon: Exploration rate for epsilon-greedy selection (0.0-1.0).
        epsilon_decay: Decay rate for epsilon over iterations (0.0-1.0).
        min_epsilon: Minimum epsilon value.

        # Acceptance criteria
        accept_equal: Accept solutions with equal quality.
        accept_worse_prob: Initial probability of accepting worse solutions.
        acceptance_decay: Decay rate for acceptance probability.

        # Normalization parameters
        iri_window: Window size for IRI normalization.
        tbi_window: Window size for TBI normalization.

        # Multi-start
        restarts: Number of random restarts.
        restart_threshold: Iterations without improvement before restart.

        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    time_limit: float = 60.0
    seed: Optional[int] = None
    max_iterations: int = 1000
    vrpp: bool = True

    # Operators
    move_operators: List[str] = field(
        default_factory=lambda: [
            "swap_intra",
            "relocate_intra",
            "two_opt_intra",
            "swap_inter",
            "relocate_inter",
            "two_opt_star",
            "exchange_10",
            "exchange_11",
            "exchange_21",
        ]
    )
    perturbation_operators: List[str] = field(
        default_factory=lambda: [
            "random_removal",
            "string_removal",
            "route_removal",
        ]
    )

    # Guidance Indicator weights
    iri_weight: float = 0.6
    tbi_weight: float = 0.4

    # Learning parameters
    learning_rate: float = 0.1
    memory_size: int = 50
    epsilon: float = 0.2
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    # Acceptance criteria
    accept_equal: bool = True
    accept_worse_prob: float = 0.05
    acceptance_decay: float = 0.99

    # Normalization
    iri_window: int = 20
    tbi_window: int = 20

    # Multi-start
    restarts: int = 1
    restart_threshold: int = 100

    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
