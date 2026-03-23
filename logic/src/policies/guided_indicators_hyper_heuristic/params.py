"""
Parameters for GIHH (Hyper-Heuristic with Two Guidance Indicators).

This module defines the configuration parameters for the GIHH algorithm.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GIHHParams:
    """
    Configuration parameters for GIHH algorithm.

    Attributes:
        time_limit (float): Maximum execution time in seconds.
        max_iterations (int): Maximum number of iterations.
        seed (Optional[int]): Random seed for reproducibility.

        # Low-level heuristic operators
        move_operators (List[str]): List of move operator names.
        perturbation_operators (List[str]): List of perturbation operator names.

        # Guidance Indicator weights
        iri_weight (float): Weight for Improvement Rate Indicator (0.0-1.0).
        tbi_weight (float): Weight for Time-based Indicator (0.0-1.0).

        # Learning parameters
        learning_rate (float): Rate of indicator updates (0.0-1.0).
        memory_size (int): Number of recent iterations to track.
        epsilon (float): Exploration rate for epsilon-greedy selection (0.0-1.0).
        epsilon_decay (float): Decay rate for epsilon over iterations (0.0-1.0).
        min_epsilon (float): Minimum epsilon value.

        # Acceptance criteria
        accept_equal (bool): Accept solutions with equal quality.
        accept_worse_prob (float): Initial probability of accepting worse solutions.
        acceptance_decay (float): Decay rate for acceptance probability.

        # Normalization parameters
        iri_window (int): Window size for IRI normalization.
        tbi_window (int): Window size for TBI normalization.

        # Multi-start
        restarts (int): Number of random restarts.
        restart_threshold (int): Iterations without improvement before restart.
    """

    # Core parameters
    time_limit: float = 60.0
    max_iterations: int = 1000
    seed: Optional[int] = None

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

    # Profit-awareness
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "GIHHParams":
        """Create parameters from a configuration object."""
        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            max_iterations=getattr(config, "max_iterations", 1000),
            seed=getattr(config, "seed", None),
            move_operators=getattr(config, "move_operators", None),
            perturbation_operators=getattr(config, "perturbation_operators", None),
            iri_weight=getattr(config, "iri_weight", 0.6),
            tbi_weight=getattr(config, "tbi_weight", 0.4),
            learning_rate=getattr(config, "learning_rate", 0.1),
            memory_size=getattr(config, "memory_size", 50),
            epsilon=getattr(config, "epsilon", 0.2),
            epsilon_decay=getattr(config, "epsilon_decay", 0.995),
            min_epsilon=getattr(config, "min_epsilon", 0.01),
            accept_equal=getattr(config, "accept_equal", True),
            accept_worse_prob=getattr(config, "accept_worse_prob", 0.05),
            acceptance_decay=getattr(config, "acceptance_decay", 0.99),
            iri_window=getattr(config, "iri_window", 20),
            tbi_window=getattr(config, "tbi_window", 20),
            restarts=getattr(config, "restarts", 1),
            restart_threshold=getattr(config, "restart_threshold", 100),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
