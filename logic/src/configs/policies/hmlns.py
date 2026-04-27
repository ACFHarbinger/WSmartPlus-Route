"""
Memetic Island Model GA configuration.

Replaces HVPL with rigorous GA terminology.

Attributes:
    HybridMemeticLargeNeighborhoodSearchConfig: Configuration for the HMLNS (Hybrid Memetic Large Neighborhood Search) policy.

Example:
    >>> from configs.policies.hmlns import HybridMemeticLargeNeighborhoodSearchConfig
    >>> config = HybridMemeticLargeNeighborhoodSearchConfig()
    >>> config.time_limit
    300.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HybridMemeticLargeNeighborhoodSearchConfig:
    """Configuration for Hybrid Memetic Large Neighborhood Search (HMLNS) policy.

    Functionally equivalent to HVPL.

    Attributes:
        population_size (int): Population size for GA.
        max_generations (int): Maximum number of generations.
        substitution_rate (float): Rate of substitution operator.
        crossover_rate (float): Rate of crossover operator.
        mutation_rate (float): Rate of mutation operator.
        elitism_count (int): Number of elite individuals.
        aco_init_iterations (int): Number of ACO initialization iterations.
        time_limit (float): Time limit in seconds.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        seed (Optional[int]): Seed for the random number generator.
        alns (dict): ALNS parameters.
        maco (dict): MACO parameters.
        mandatory_selection (Optional[List[MandatorySelectionConfig]]): Mandatory selection configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    population_size: int = 30
    max_generations: int = 50
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 3
    aco_init_iterations: int = 50
    time_limit: float = 300.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    alns: dict = field(default_factory=dict)
    maco: dict = field(default_factory=dict)
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
