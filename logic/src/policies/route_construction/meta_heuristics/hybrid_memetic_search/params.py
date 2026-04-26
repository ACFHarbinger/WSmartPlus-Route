"""
Parameters for the Hybrid Memetic Search (HMS).

Attributes:
    HybridMemeticSearchParams: Configuration dataclass for HMS.

Example:
    >>> params = HybridMemeticSearchParams(population_size=50)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import (
    ALNSParams,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)


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
        n_removal: Number of nodes to remove in mutation.
        aco_init_iterations: ACO iterations for population seeding.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether to solve the VRP with profits.
        profit_aware_operators: Whether to use profit-aware heuristics.
        aco_params: ACO configuration.
        alns_params: ALNS configuration.
    """

    # Population Parameters
    population_size: int = 30
    max_generations: int = 100
    substitution_rate: float = 0.2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 3

    # Phase-specific Parameters
    n_removal: int = 3
    aco_init_iterations: int = 50
    time_limit: float = 300.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Sub-algorithm Parameters
    aco_params: Optional[KSACOParams] = field(default_factory=lambda: None)
    alns_params: Optional[ALNSParams] = field(default_factory=lambda: None)

    def __post_init__(self):
        """Initialize sub-algorithm parameters with defaults if not provided.

        Args:
            None.

        Returns:
            None.
        """
        if self.aco_params is None:
            self.aco_params = KSACOParams(
                n_ants=20,
                k_sparse=10,
                alpha=1.0,
                beta=2.0,
                rho=0.1,
                scale=5.0,
                tau_0=None,
                tau_min=0.001,
                tau_max=10.0,
                max_iterations=1,
                time_limit=60.0,
                local_search=False,
                vrpp=self.vrpp,
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators

        if self.alns_params is None:
            self.alns_params = ALNSParams(
                max_iterations=100,
                start_temp=100.0,
                cooling_rate=0.95,
                reaction_factor=0.1,
                min_removal=1,
                start_temp_control=100.0,
                xi=1.0,
                segment_size=10,
                time_limit=60.0,
                vrpp=self.vrpp,
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators

    @classmethod
    def from_config(cls, config: Any) -> "HybridMemeticSearchParams":
        """Create HybridMemeticSearchParams from an HMSConfig dataclass or dict.

        Args:
            config: Configuration source (dataclass or dictionary).

        Returns:
            HybridMemeticSearchParams: Initialized parameter object.
        """
        if isinstance(config, dict):
            return cls(
                population_size=config.get("population_size", 30),
                max_generations=config.get("max_generations", 100),
                substitution_rate=config.get("substitution_rate", 0.2),
                crossover_rate=config.get("crossover_rate", 0.8),
                mutation_rate=config.get("mutation_rate", 0.1),
                elitism_count=config.get("elitism_count", 3),
                aco_init_iterations=config.get("aco_init_iterations", 50),
                time_limit=config.get("time_limit", 300.0),
                vrpp=config.get("vrpp", True),
                n_removal=config.get("n_removal", 3),
                profit_aware_operators=config.get("profit_aware_operators", False),
                aco_params=KSACOParams.from_config(config.get("aco")) if config.get("aco") else None,  # type: ignore[arg-type]
                alns_params=ALNSParams.from_config(config.get("alns")) if config.get("alns") else None,
            )

        return cls(
            population_size=config.population_size,
            max_generations=config.max_generations,
            substitution_rate=config.substitution_rate,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            elitism_count=config.elitism_count,
            aco_init_iterations=config.aco_init_iterations,
            time_limit=config.time_limit,
            n_removal=config.n_removal,
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            aco_params=KSACOParams.from_config(config.aco) if config.aco else None,
            alns_params=ALNSParams.from_config(config.alns) if config.alns else None,
        )

    # ------------------------------------------------------------------
    # Compatibility aliases for EXACT matching with HVPL attribute names
    # ------------------------------------------------------------------

    @property
    def max_iterations(self) -> int:
        """Alias for max_generations to match HVPL exactly.

        Args:
            None.

        Returns:
            int: Total generations.
        """
        return self.max_generations
