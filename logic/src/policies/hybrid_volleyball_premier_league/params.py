"""
Hyperparameters for Hybrid Volleyball Premier League (HVPL) algorithm.
"""

from dataclasses import dataclass, field

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams


@dataclass
class HVPLParams:
    """
    Parameters for the Hybrid Volleyball Premier League metaheuristic.
    """

    # VPL Population Dynamics
    n_teams: int = 10  # Population size
    max_iterations: int = 50  # Number of league seasons
    sub_rate: float = 0.2  # Fraction of teams replaced by substitution
    time_limit: float = 60.0  # Overall time limit in seconds

    # ACO Components (Initialization & Global Guidance)
    aco_params: ACOParams = field(
        default_factory=lambda: ACOParams(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Only one iteration per construction phase
            time_limit=30,
            local_search=False,  # ALNS handles local search
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )

    # ALNS Components (Coaching & Improvement)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=100,  # "Coaching session" length
            start_temp=100.0,
            cooling_rate=0.95,
            reaction_factor=0.5,
            min_removal=1,
            max_removal_pct=0.2,
            time_limit=30,
        )
    )
