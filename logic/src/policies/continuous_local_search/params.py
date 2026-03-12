"""
Configuration parameters for Continuous Local Search.

This replaces the metaphor-based "Sine Cosine Algorithm" with rigorous terminology.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContinuousLocalSearchParams:
    """
    Parameters for Continuous Local Search with trigonometric perturbations.

    Standard continuous optimization with trigonometric step functions for
    exploration/exploitation balance. Replaces the "Sine Cosine Algorithm" metaphor:
    - "Destination point" → Best solution
    - "Position update equation" → Gradient-free perturbation operator
    - "Sine/Cosine switching" → Randomized directional search
    - "Parameter a" → Step size decay schedule

    Algorithm:
        1. Encode solutions as continuous vectors
        2. Apply trigonometric perturbations: x'[i] = x[i] + α × sin/cos(θ) × |β × x_best[i] - x[i]|
        3. Decay step size α linearly from α_max to 0
        4. Decode continuous vectors to discrete routing solutions

    Attributes:
        population_size: Number of solution vectors in population.
        max_step_size: Initial perturbation step size (α_max).
        max_iterations: Maximum number of search iterations.
        time_limit: Wall-clock time limit in seconds (0 = no limit).

    Complexity:
        - Space: O(pop_size × n) for continuous population
        - Time per iteration: O(pop_size × n) for perturbation + O(n²) for decoding

    Mathematical Foundation:
        Position update: x'[i] = x[i] + r₁ × sin(r₂) × |r₃ × x_best[i] - x[i]|
        or: x'[i] = x[i] + r₁ × cos(r₂) × |r₃ × x_best[i] - x[i]|
        where r₁ ∈ [0, α], r₂ ∈ [0, 2π], r₃ ∈ [0, 2], α decays linearly
    """

    population_size: int = 30
    max_step_size: float = 2.0  # α_max in update equations
    max_iterations: int = 500
    time_limit: float = 60.0
