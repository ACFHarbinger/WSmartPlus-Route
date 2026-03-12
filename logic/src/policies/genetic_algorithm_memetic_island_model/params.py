"""
Configuration parameters for Island Model Genetic Algorithm.

This replaces metaphor-based sports algorithms (HVPL, LCA, SLC) with
rigorous terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..adaptive_large_neighborhood_search.params import ALNSParams
from ..ant_colony_optimization.k_sparse_aco.params import ACOParams


@dataclass
class MemeticIslandModelGAParams:
    """
    Parameters for Island Model Genetic Algorithm with Tournament Selection.

    Multi-population evolutionary algorithm where separate sub-populations (islands)
    evolve independently with periodic migration. Replaces sports metaphors:
    - "Teams" → Islands (sub-populations)
    - "Matches" → Fitness evaluations
    - "Seasons" → Generations
    - "Coaching" → Local search operator
    - "Relegation/Promotion" → Population replacement via tournament selection

    Algorithm Structure:
        1. Initialize K islands with N individuals each
        2. For each generation:
            a. Local improvement: Apply optimization to each island
            b. Fitness evaluation: Assess all individuals
            c. Migration: Exchange best solutions between islands
            d. Replacement: Use tournament selection to replace weakest individuals

    Attributes:
        n_islands: Number of independent sub-populations (K).
        island_size: Population size per island (N).
        max_generations: Number of evolution cycles.
        replacement_rate: Fraction of population replaced per generation.
        tournament_size: Number of individuals competing in tournament selection.
        migration_interval: Generations between inter-island migration.
        migration_size: Number of individuals migrated per interval.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
        aco_params: ACO parameters for constructive heuristic.
        alns_params: ALNS parameters for local search improvement.

    Complexity:
        - Space: O(K × N × n) for island populations
        - Time per generation: O(K × N × local_search_cost)

    Reference:
        Whitley, D., Rana, S., & Heckendorn, R. B. (1998).
        "The island model genetic algorithm: On separability, population size and convergence."
        Journal of Computing and Information Technology.
    """

    # Island topology
    n_islands: int = 10  # K parameter
    island_size: int = 10  # N parameter (individuals per island)
    max_generations: int = 50
    time_limit: float = 60.0

    # Selection and replacement
    replacement_rate: float = 0.2  # Fraction replaced per generation
    tournament_size: int = 3  # Tournament selection pressure

    # Migration parameters
    migration_interval: int = 5  # Generations between migrations
    migration_size: int = 1  # Best solutions migrated per interval

    # Constructive heuristic (ACO for initialization and diversity)
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
            max_iterations=1,
            time_limit=30,
            local_search=False,
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )

    # Local search operator (ALNS for improvement)
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=100,
            start_temp=100.0,
            cooling_rate=0.95,
            reaction_factor=0.5,
            min_removal=1,
            max_removal_pct=0.2,
            time_limit=30,
        )
    )
