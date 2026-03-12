"""
HGS (Hybrid Genetic Search) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class HGSConfig:
    """Configuration for Hybrid Genetic Search (HGS) policy.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        time_limit: Maximum time in seconds for the solver (0.0 = unlimited).
        seed: Random seed for reproducibility.
        mu: Minimum population size per subpopulation.
        n_offspring: Generation size (number before survivor selection).
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        nb_granular: Granular search parameter for local search.
        target_feasible: Target proportion of feasible solutions.
        n_iterations_no_improvement: Max iterations without improvement.
        mutation_rate: Probability of applying local search to offspring.
        repair_probability: Probability of repairing infeasible offspring.
        crossover_rate: Probability of applying crossover.
        alpha_diversity: Weight for diversity in fitness evaluation.
        min_diversity: Minimum diversity threshold.
        diversity_change_rate: Rate at which alpha diversity changes.
        local_search_iterations: Number of local search iterations.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        initial_penalty_capacity: Initial penalty for capacity violations.
        penalty_increase: Multiplier for increasing penalty.
        penalty_decrease: Multiplier for decreasing penalty.
        engine: Solver engine to use ('custom', 'pyvrp').
        vrpp: Whether this is a VRPP problem.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    # Core HGS parameters (Vidal 2022)
    time_limit: float = 0.0
    seed: Optional[int] = None
    mu: int = 25
    n_offspring: int = 40
    nb_elite: int = 4
    nb_close: int = 5
    nb_granular: int = 20
    target_feasible: float = 0.2
    n_iterations_no_improvement: int = 20000

    # Genetic operators
    mutation_rate: float = 1.0
    repair_probability: float = 0.5
    crossover_rate: float = 1.0

    # Diversity management
    alpha_diversity: float = 0.5
    min_diversity: float = 0.2
    diversity_change_rate: float = 0.05

    # Local search
    local_search_iterations: int = 100
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85

    # Engine configuration
    engine: str = "custom"
    vrpp: bool = True
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
