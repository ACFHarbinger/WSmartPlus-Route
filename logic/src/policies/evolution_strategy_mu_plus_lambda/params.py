"""
Configuration parameters for the (μ+λ) Evolution Strategy solver.

This replaces the metaphor-based "Harmony Search" with rigorous terminology.

IMPORTANT: To exactly match Harmony Search, set λ=1 (offspring_size=1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuPlusLambdaESParams:
    """
    Parameters for (μ+λ) Evolution Strategy.

    A (μ+λ)-ES maintains a population of μ parent solutions. Each iteration:
    1. Select parents to create λ offspring.
    2. Create offspring via recombination (using archive solutions).
    3. Apply mutation operator (local perturbation).
    4. Combine parents and offspring (μ+λ individuals).
    5. Select the best μ individuals to survive to the next generation.

    This is the canonical interpretation of what "Harmony Search" actually implements
    when λ=1. The Harmony Search algorithm generates exactly 1 new harmony per
    iteration and replaces the worst if better, which is equivalent to (μ+1)-ES.

    Attributes:
        population_size (μ): Number of parent solutions in archive.
                             Equivalent to Harmony Search "hm_size".
        offspring_size (λ): Number of offspring generated per iteration.
                           Set to 1 for exact Harmony Search equivalence.
        recombination_rate: Probability of using archive solutions (vs. random).
                           Equivalent to Harmony Search "HMCR".
        mutation_rate: Probability of applying local mutation after recombination.
                      Equivalent to Harmony Search "PAR".
        max_iterations: Maximum number of evolution cycles.
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(μ × n) for population storage
        - Time per iteration: O(λ × n²) for offspring generation and evaluation

    Note:
        The default λ=1 ensures exact equivalence with Harmony Search.
        Larger λ values create a generalized (μ+λ)-ES with faster convergence
        but may reduce diversity compared to the original HS algorithm.
    """

    population_size: int = 10  # μ parameter (equivalent to hm_size)
    offspring_size: int = 1  # λ parameter
    recombination_rate: float = 0.95  # Equivalent to HMCR
    mutation_rate: float = 0.3  # Equivalent to PAR
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
