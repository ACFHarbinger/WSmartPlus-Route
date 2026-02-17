"""
AHVPL (Augmented Hybrid Volleyball Premier League) configuration for expert policy training.
"""

from dataclasses import dataclass


@dataclass
class AHVPLConfig:
    """Configuration for Augmented Hybrid Volleyball Premier League (AHVPL) expert policy.

    Extends HVPL with HGS integration parameters for diversity-driven
    crossover and bi-criteria fitness evaluation.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of league iterations.
        n_teams: Number of teams (solutions) per instance in the league.
        sub_rate: Rate of substitution for weak teams each iteration.
        aco_iterations: Number of ACO iterations for initialization/substitution.
        alns_iterations: Number of ALNS iterations for the coaching phase.
        hgs_elite_size: Number of elite solutions for biased fitness ranking.
        hgs_mutation_rate: Probability of HGS local search mutation.
        hgs_crossover_rate: Probability of applying OX crossover.
        hgs_max_vehicles: Maximum vehicles allowed (0 = unlimited).
        hgs_population_size: HGS internal population size.
    """

    time_limit: float = 60.0
    max_iterations: int = 50
    n_teams: int = 10
    sub_rate: float = 0.2
    aco_iterations: int = 1
    alns_iterations: int = 100
    hgs_elite_size: int = 5
    hgs_mutation_rate: float = 0.2
    hgs_crossover_rate: float = 0.7
    hgs_max_vehicles: int = 0
    hgs_population_size: int = 50
