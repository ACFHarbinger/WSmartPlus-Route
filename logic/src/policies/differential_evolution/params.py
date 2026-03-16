"""
Configuration parameters for the Differential Evolution (DE) solver.

This module defines the structural constraints and hyper-parameters required to
instantiate a Differential Evolution algorithm, following the rigorous formulation
by Storn & Price (1997).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DEParams:
    """
    Configuration parameters for Differential Evolution (DE/rand/1/bin).

    Differential Evolution is a population-based evolutionary algorithm that uses
    vector differences for mutation and binomial crossover for recombination.
    Unlike ABC (which is DE with fitness-proportionate selection), DE uses
    greedy one-to-one selection.

    Attributes:
        pop_size (int): Population size (NP). Number of candidate solution vectors.
            Standard recommendations: NP = 10×D where D is dimensionality.
            For routing: NP typically 20-100.

        mutation_factor (float): Differential weight (F). Scaling factor for the
            mutation vector. Controls the amplification of the differential variation.
            Range: [0, 2]. Standard: F ∈ [0.5, 1.0].
            - F = 0.5: Conservative exploration
            - F = 1.0: Aggressive exploration (classical DE)

        crossover_rate (float): Crossover probability (CR). Probability that a
            component is inherited from the mutant vector rather than the target.
            Range: [0, 1]. Standard: CR ∈ [0.8, 0.95].
            - CR = 0.0: No recombination (pure mutation)
            - CR = 1.0: Full replacement (no inheritance)

        n_removal (int): Mutation strength parameter for discrete operators.
            Number of nodes removed during destroy-repair mutation. This acts as
            the discrete analog to continuous mutation magnitude.

        max_iterations (int): Maximum number of generations (G_max).
            Primary termination criterion for the evolutionary loop.

        local_search_iterations (int): Intensity of local optimization applied
            to each trial vector. Governs the fine-tuning of candidate solutions
            post-mutation.

        time_limit (float): Wall-clock time limit in seconds. Algorithm terminates
            early if process time exceeds this threshold.

    Mathematical Foundation:
        DE/rand/1/bin strategy:

        1. Mutation:
           v_i = x_r1 + F × (x_r2 - x_r3)
           where r1, r2, r3 are distinct random indices ≠ i

        2. Crossover (Binomial):
           u_ij = v_ij  if rand() < CR or j = j_rand
                  x_ij  otherwise

        3. Selection (Greedy):
           x_i(t+1) = u_i  if f(u_i) ≥ f(x_i)
                      x_i  otherwise

    Reference:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.
    """

    pop_size: int = 50  # Population size (NP)
    mutation_factor: float = 0.8  # Differential weight (F)
    crossover_rate: float = 0.9  # Crossover probability (CR)
    n_removal: int = 3  # Mutation strength for discrete operators
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
