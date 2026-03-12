"""
Configuration parameters for Stochastic Tournament Genetic Algorithm.

This replaces the metaphor-based "League Championship Algorithm (LCA)".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StochasticTournamentGAParams:
    """
    Parameters for Genetic Algorithm with Stochastic Tournament Selection.

    Canonical GA where selection pressure is applied via pairwise stochastic
    tournaments. Each individual competes directly against others, with
    selection probability proportional to fitness difference.

    Replaces "League Championship Algorithm" sports metaphors:
    - "League Schedule/Fixtures" → Pairwise fitness evaluation cycles
    - "Playing Strength" → Objective function value (fitness score)
    - "Match Outcome (Win/Loss)" → Stochastic tournament result
    - "Team Formation" → Recombination/crossover operator

    Algorithm Structure:
        1. Initialize population of N chromosomes
        2. For each generation:
            a. Fitness evaluation for all individuals
            b. Pairwise stochastic tournament selection
            c. Crossover (recombination) of selected parents
            d. Mutation of offspring
            e. Elitist replacement (keep best individuals)

    Attributes:
        population_size: Number of chromosomes in population (N).
        tournament_competitors: Number of opponents per individual in tournament.
        selection_pressure: Controls stochastic tournament probability.
            P(select i over j) = 1 / (1 + exp(-pressure × (fitness_i - fitness_j)))
        crossover_rate: Probability of applying crossover operator.
        mutation_rate: Probability of applying mutation operator.
        elitism_count: Number of top individuals preserved unchanged.
        max_generations: Maximum number of evolution cycles.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(N × n) for population storage
        - Time per generation: O(N × tournament_size × eval_cost)

    Mathematical Foundation:
        Stochastic tournament selection with sigmoid probability:
        P(i defeats j) = σ(β × (f(i) - f(j)))
        where σ(x) = 1/(1 + exp(-x)), β = selection_pressure
    """

    population_size: int = 50  # N parameter
    tournament_competitors: int = 5  # Opponents per individual
    selection_pressure: float = 0.1  # β coefficient for sigmoid
    crossover_rate: float = 0.8  # Probability of recombination
    mutation_rate: float = 0.2  # Probability of perturbation
    elitism_count: int = 2  # Top individuals preserved
    max_generations: int = 100
    time_limit: float = 60.0
