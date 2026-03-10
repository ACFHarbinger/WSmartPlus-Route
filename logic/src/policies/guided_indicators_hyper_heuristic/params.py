"""
Parameters for GIHH (Hyper-Heuristic with Two Guidance Indicators).

This module defines the configuration parameters for the GIHH algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Optional


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
