"""
Hybrid Memetic Search (HMS) Parameters.

This module defines the configuration parameters for the Hybrid Memetic
Search algorithm, which is a triple-phase hybrid solver combining ACO,
Genetic Algorithms (HGS variant), and ALNS.
"""

from dataclasses import dataclass, field

from ..adaptive_large_neighborhood_search import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams


@dataclass
class HybridMemeticSearchParams:
    """
    Configuration parameters for Hybrid Memetic Search (HMS).

    Functionally identical to HVPL but with rigorous nomenclature.

    Attributes:
        population_size: Number of chromosomes in the active population.
        max_generations: Total number of evolutionary generations.
        substitution_rate: Fraction of population replaced by reserve pool.
        crossover_rate: Probability of recombining two parents.
        mutation_rate: Probability of local perturbation.
        elitism_count: Number of best solutions to preserve across generations.
        aco_init_iterations: ACO iterations for population seeding.
        alns_iterations: ALNS iterations per refinement session.
        time_limit: Wall-clock time limit in seconds.
        aco_params: ACO configuration.
        alns_params: ALNS configuration.
    """

    # Population Parameters
    population_size: int = 30
    max_generations: int = 50
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 3

    # Phase-specific Parameters
    aco_init_iterations: int = 50
    alns_iterations: int = 100
    time_limit: float = 300.0

    # Sub-algorithm Parameters
    aco_params: ACOParams = field(
        default_factory=lambda: ACOParams(
            n_ants=20,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,
            time_limit=30,
            local_search=False,
        )
    )

    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=100,
            start_temp=100.0,
            cooling_rate=0.95,
            time_limit=30,
        )
    )
