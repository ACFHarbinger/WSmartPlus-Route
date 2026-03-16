"""
Configuration parameters for (μ,λ) Evolution Strategy.

This module defines the structural constraints and hyper-parameters required to
instantiate a generational Evolution Strategy. It follows the standard notation
where μ represents the parent population and λ represents the offspring population.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuCommaLambdaESParams:
    """
    Parameters for (μ,λ) Evolution Strategy with deterministic truncation selection.

    A (μ,λ)-ES maintains a parent population of size μ and generates λ offspring
    each iteration. This configuration ensures a memoryless stochastic process
    (Markov process), where the next parent population is derived strictly from
    the current offspring pool.

    Attributes:
        mu (int): The number of parent individuals maintained in the population (μ).
            A larger μ improves collective hillclimbing and reliability in
            multi-modal landscapes.
            Default follows standard big population settings.

        lambda_ (int): The number of offspring generated in each generation (λ).
            Standard (μ,λ) theory suggests that λ should be significantly
            larger than μ (e.g., λ ≈ 7μ) to ensure efficient step-size
            self-adaptation.

        n_removal (int): The mutation strength parameter. Defines the number
            of nodes removed during the destroy-repair mutation phase. This acts
            as the random perturbation added to components of the candidate
            solution.

        max_iterations (int): The generational loop limit. Serves as a
            primary termination criterion for the search process.

        local_search_iterations (int): The intensity of the local optimization
            applied to each offspring. This governs the fine-tuning of
            candidate solutions post-mutation.

        time_limit (float): Wall-clock duration in seconds. The algorithm
            will terminate early if the process time exceeds this threshold
            to ensure compliance with real-time operational constraints.

    Mathematical Foundation:
        1. Variation: λ offspring are produced through recombination (averaging
           or discrete selection) and mutation.
        2. Selection: The parent population P_{t+1} is formed by selecting the
           absolute best μ individuals from the λ offspring pool.
    """

    mu: int = 15  # Parent population size (μ)
    lambda_: int = 100  # Offspring population size (λ)
    n_removal: int = 3  # Mutation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
