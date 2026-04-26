r"""Hyperparameters for the Augmented Hybrid Volleyball Premier League (AHVPL) algorithm.

Extends the base HVPL framework with Hybrid Genetic Search (HGS) integration
for diversity-driven population management and genetic crossover operators.

Attributes:
    AHVPLParams: Parameters for the Augmented Hybrid Volleyball Premier League metaheuristic.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params import AHVPLParams
    >>> params = AHVPLParams(n_teams=12, max_iterations=100)
"""

from dataclasses import dataclass, field
from typing import Optional

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
    ALNSParams,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams


@dataclass
class AHVPLParams:
    """
    Parameters for the Augmented Hybrid Volleyball Premier League metaheuristic.

    Combines VPL population dynamics, ACO initialization, ALNS local search,
    and HGS diversity management / crossover.

    Attributes:
        n_teams: Number of teams in the league.
        max_iterations: Maximum number of league seasons.
        sub_rate: Probability of player substitution.
        elite_alns_iterations: ALNS iterations for coached teams.
        not_coached_alns_iterations: ALNS iterations for non-coached teams.
        time_limit: Total wall-clock time limit.
        vrpp: Whether to solve as a VRP with Profits.
        seed: Random seed for reproducibility.
        profit_aware_operators: Whether to use operators biased towards profit.
        hgs_params: Parameters for Hybrid Genetic Search components.
        aco_params: Parameters for Ant Colony Optimization components.
        alns_params: Parameters for Adaptive Large Neighborhood Search components.
    """

    # VPL Population Dynamics
    n_teams: int = 10
    max_iterations: int = 50
    sub_rate: float = 0.2
    elite_alns_iterations: int = 500
    not_coached_alns_iterations: int = 100
    time_limit: float = 60.0
    vrpp: bool = True
    seed: Optional[int] = None
    profit_aware_operators: bool = False

    # HGS Components (Diversity Management & Crossover)
    hgs_params: HGSParams = field(
        default_factory=lambda: HGSParams(
            nb_elite=5,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_vehicles=0,
            mu=50,
            n_offspring=40,  # Default for lambda_param
            n_iterations_no_improvement=10,
            nb_granular=10,
            local_search_iterations=100,
            time_limit=30.0,
            vrpp=True,
            profit_aware_operators=False,
        )
    )

    # ACO Components (Initialization & Global Guidance)
    aco_params: KSACOParams = field(
        default_factory=lambda: KSACOParams(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            scale=5.0,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,
            time_limit=30.0,
            local_search=False,
            local_search_iterations=0,
            elitist_weight=1.0,
            vrpp=True,
            profit_aware_operators=False,
        )
    )

    # ALNS Components (Coaching & Deep Local Search)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            time_limit=30.0,
            max_iterations=100,
            start_temp=100.0,
            cooling_rate=0.95,
            reaction_factor=0.1,
            min_removal=1,
            max_removal_pct=0.2,
            vrpp=True,
            profit_aware_operators=False,
        )
    )

    def __post_init__(self):
        """Sync flags across sub-parameters.

        Returns:
            None.
        """
        if self.aco_params:
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators
        if self.alns_params:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators
        if self.hgs_params:
            self.hgs_params.vrpp = self.vrpp
            self.hgs_params.profit_aware_operators = self.profit_aware_operators
