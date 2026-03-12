"""
Configuration parameters for (μ,λ) Evolution Strategy.

This replaces the metaphor-based "Artificial Bee Colony" with rigorous terminology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuCommaLambdaESParams:
    """
    Parameters for (μ,λ) Evolution Strategy with multi-phase random search.

    A (μ,λ)-ES maintains μ parents and generates λ offspring each iteration:
    1. **Local Search Phase**: Each parent generates offspring via localized mutation
    2. **Probabilistic Selection Phase**: Offspring compete via fitness-proportional selection
    3. **Random Restart Phase**: Stagnant solutions are replaced with new random samples

    This is the canonical interpretation of what "Artificial Bee Colony" actually implements
    when stripped of bee metaphors:
    - "Employed bees" → Local search agents (exploitation)
    - "Onlooker bees" → Probabilistic selection mechanism
    - "Scout bees" → Random restart mechanism (exploration)

    Attributes:
        population_size (μ): Number of parent solutions maintained.
        offspring_per_parent (λ/μ): Number of offspring per parent.
        n_removal: Perturbation strength (nodes removed during mutation).
        stagnation_limit: Restart threshold for solutions without improvement.
        max_iterations: Maximum number of evolution cycles.
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(μ × n) for parent storage + O(λ × n) for offspring
        - Time per iteration: O(λ × n²) for offspring generation and evaluation

    Mathematical Foundation:
        Each iteration:
        1. Generate λ offspring from μ parents (λ ≥ μ)
        2. Select best μ from offspring (non-elitist: parents discarded)
        3. Restart stagnant solutions after limit iterations without improvement
    """

    population_size: int = 20  # μ parameter
    offspring_per_parent: int = 1  # λ = μ × offspring_per_parent
    n_removal: int = 3  # Mutation strength
    stagnation_limit: int = 10  # Random restart threshold
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
