"""
Parameters for Hybrid Genetic Search with Ruin-and-Recreate (HGSRR).

This module defines the configuration parameters for the HGSRR algorithm,
extending standard HGS with adaptive destroy/repair operator management.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class HGSRRParams:
    """
    Configuration parameters for HGSRR algorithm.

    Attributes:
        time_limit (float): Maximum execution time in seconds.
        mu (int): Minimum population size (for each subpopulation).
        nb_elite (int): Number of elite individuals preserved.
        mutation_rate (float): Probability of applying ruin-recreate mutation.
        n_offspring (int): Generation size (individuals created per iteration).
        alpha_diversity (float): Initial diversity weight in fitness calculation.
        min_diversity (float): Minimum population diversity threshold.
        diversity_change_rate (float): Rate of alpha adjustment.
        n_iterations_no_improvement (int): Max iterations without improvement.
        survivor_threshold (float): Population size multiplier for survivor selection.
        max_vehicles (int): Maximum number of vehicles (0 = unlimited).
        crossover_rate (float): Probability of crossover operation.
        nb_granular (int): Granular search parameter for local search.

        # Ruin-and-Recreate specific parameters
        min_removal_pct (float): Minimum percentage of nodes to remove (0.0-1.0).
        max_removal_pct (float): Maximum percentage of nodes to remove (0.0-1.0).
        noise_factor (float): Noise factor for randomized best insertion.
        reaction_factor (float): Rate of operator weight updates.
        decay_parameter (float): Decay rate for operator scores.

        # Operator selection weights (initial)
        destroy_operators (List[str]): List of destroy operator names.
        repair_operators (List[str]): List of repair operator names.
        operator_decay_rate (float): Exponential decay rate for operator weights.

        # Scoring parameters for adaptive selection
        score_sigma_1 (float): Score for new global best solution.
        score_sigma_2 (float): Score for solution improving current.
        score_sigma_3 (float): Score for solution accepted but not improving.

        seed (Optional[int]): Random seed for reproducibility.
    """

    # Core HGS parameters
    time_limit: float = 10.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.3
    n_offspring: int = 40  # Generation size
    n_generations: int = 100  # Total iterations (preserved as is)
    alpha_diversity: float = 0.5
    min_diversity: float = 0.2
    diversity_change_rate: float = 0.05
    no_improvement_threshold: int = 20  # Threshold for diversity/stopping
    survivor_threshold: float = 2.0
    max_vehicles: int = 0
    crossover_rate: float = 0.7
    neighbor_list_size: int = 10

    # Ruin-and-Recreate parameters
    min_removal_pct: float = 0.1
    max_removal_pct: float = 0.4
    noise_factor: float = 0.015
    reaction_factor: float = 0.1
    decay_parameter: float = 0.95

    # Operator management
    destroy_operators: List[str] = field(
        default_factory=lambda: [
            "random_removal",
            "worst_removal",
            "cluster_removal",
            "shaw_removal",
            "string_removal",
        ]
    )
    repair_operators: List[str] = field(
        default_factory=lambda: [
            "greedy_insertion",
            "regret_2_insertion",
            "regret_k_insertion",
            "greedy_insertion_with_blinks",
        ]
    )
    operator_decay_rate: float = 0.95

    # Adaptive scoring
    score_sigma_1: float = 33.0  # New global best
    score_sigma_2: float = 9.0  # Improvement
    score_sigma_3: float = 3.0  # Accepted

    seed: Optional[int] = None

    seed: Optional[int] = None
