"""
Hyperparameters for the Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS) algorithm.

Combines HGS evolutionary operators with ALNS-based education phase for intensive local search.
"""

from dataclasses import dataclass, field

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..hybrid_genetic_search.params import HGSParams


@dataclass
class HGSALNSParams:
    """
    Parameters for the Hybrid Genetic Search with ALNS Education metaheuristic.

    Uses ALNS for the education phase and HGS for the routing phase, combining
    the exploration power of genetic algorithms with the exploitation strength
    of adaptive large neighborhood search.
    """

    # Hybrid-specific parameters
    time_limit: float = 60.0
    alns_education_iterations: int = 50
    hgs_max_iter: int = 100

    # HGS Components (Genetic Evolution & Population Management)
    hgs_params: HGSParams = field(
        default_factory=lambda: HGSParams(
            time_limit=60.0,
            mu=50,
            nb_elite=10,
            mutation_rate=0.2,
            crossover_rate=0.7,
            n_offspring=40,  # Default for lambda_param
            alpha_diversity=0.5,
            min_diversity=0.1,
            diversity_change_rate=0.05,
            n_iterations_no_improvement=10,
            nb_granular=10,
            local_search_iterations=500,
            max_vehicles=0,
        )
    )

    # ALNS Components (Education & Intensive Local Search)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            time_limit=60.0,
            max_iterations=50,
            start_temp=100.0,
            cooling_rate=0.995,
            reaction_factor=0.1,
            min_removal=1,
            max_removal_pct=0.3,
        )
    )
