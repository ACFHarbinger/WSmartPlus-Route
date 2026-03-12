"""
Configuration parameters for the (μ+1) Evolution Strategy solver.

This replaces the metaphor-based "Harmony Search" with rigorous terminology.
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

    This is the canonical interpretation of what "Harmony Search" actually implements.

    Attributes:
        population_size (μ): Number of parent solutions in archive.
        recombination_rate: Probability of using archive solutions (vs. random).
        mutation_rate: Probability of applying local mutation after recombination.
        max_iterations: Maximum number of evolution cycles.
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(μ × n) for population storage
        - Time per iteration: O(n) for mutation + O(n²) for evaluation
    """

    population_size: int = 10  # μ parameter
    offspring_size: int = 5  # λ parameter
    recombination_rate: float = 0.95  # Probability of archive recombination
    mutation_rate: float = 0.3  # Probability of local mutation
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
