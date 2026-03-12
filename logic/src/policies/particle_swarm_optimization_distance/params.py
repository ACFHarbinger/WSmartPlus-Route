"""
Configuration parameters for Distance-Based Particle Swarm Optimization.

This replaces the metaphor-based "Firefly Algorithm" with standard PSO terminology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistancePSOParams:
    """
    Parameters for Distance-Based Particle Swarm Optimization.

    Standard PSO with distance-dependent attraction weights. Each particle:
    1. Is attracted to global best with weight decaying exponentially by distance
    2. Performs random walk exploration with probability α_explore
    3. Updates position via destroy-repair operators

    This is what "Firefly Algorithm" actually implements when stripped of metaphor.

    Attributes:
        population_size: Number of particles in swarm.
        initial_attraction: Initial global best attraction coefficient (β₀).
        distance_decay: Exponential decay coefficient for distance-based attraction (γ).
        exploration_rate: Probability of random walk instead of global best attraction (α).
        n_removal: Number of nodes to remove during perturbation operators.
        max_iterations: Maximum number of PSO iterations.
        local_search_iterations: Number of local search improvement steps.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
        alpha_profit: Weight for profit term in attraction scoring.
        beta_will: Weight for willingness (fill level) term in attraction scoring.
        gamma_cost: Weight for insertion cost penalty in attraction scoring.

    Complexity:
        - Space: O(pop_size × n) for swarm storage
        - Time per iteration: O(pop_size² × n²) for pairwise attraction evaluation

    Mathematical Foundation:
        Attraction weight: β(d) = β₀ × exp(-γ × d²)
        where d = Hamming distance between particle solutions
    """

    population_size: int = 20
    initial_attraction: float = 1.0  # β₀ in PSO literature
    distance_decay: float = 0.01  # γ decay coefficient
    exploration_rate: float = 0.1  # α random walk probability
    n_removal: int = 3  # Perturbation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0

    # Scoring weights for guided insertion
    alpha_profit: float = 1.0
    beta_will: float = 0.5
    gamma_cost: float = 0.3
