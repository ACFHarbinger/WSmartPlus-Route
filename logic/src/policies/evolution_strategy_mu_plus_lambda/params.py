"""
Configuration parameters for the (μ+λ) Evolution Strategy.

This module adheres to the standard notation where μ represents
the parent population and λ represents the offspring population.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MuPlusLambdaESParams:
    r"""
    Parameters for the (μ+λ) Evolution Strategy.

    A (μ+λ)-ES maintains a population of μ parent solutions. In each iteration,
    it generates λ offspring through recombination and mutation.
    The selection operator then selects the absolute best μ individuals from the
    combined pool of μ parents and λ offspring to form the next generation.

    Attributes:
        mu (int): The number of parent individuals maintained in the population ($\mu$).
            This acts as an archive of the best-found solutions, guaranteeing
            strong elitism and monotonic improvement.

        lambda_ (int): The number of offspring generated per generation ($\lambda$).
            This defines the exploration capacity per cycle.

        n_removal (int): The mutation strength parameter. Defines the number
            of nodes removed during the destroy-repair mutation phase.

        max_iterations (int): The generational loop limit. Serves as a
            primary termination criterion for the search process.

        local_search_iterations (int): The intensity of the local optimization
            applied to each offspring.

        time_limit (float): Wall-clock duration in seconds. The algorithm
            will terminate early if the process time exceeds this threshold.
    """

    mu: int = 10  # Parent population size (μ)
    lambda_: int = 5  # Offspring population size (λ)
    n_removal: int = 3  # Mutation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
