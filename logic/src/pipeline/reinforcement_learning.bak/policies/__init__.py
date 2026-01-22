"""
Reinforcement Learning Policies and Heuristics.

This module contains implementations of routing policies and heuristics that can be used
either as baselines for comparison or as components within hybrid RL-OR approaches.
It covers both constructive heuristics and improvement-based metaheuristics.

Key Implementations:
- **Vectorized HGS (`hgs_vectorized.py`)**: A highly optimized, GPU-accelerated implementation
  of the Hybrid Genetic Search (HGS) algorithm.
  - Class: `VectorizedHGS`.
  - Features:
    - **Vectorized Split**: Efficiently decodes batches of giant tours into feasible routes (`vectorized_linear_split`).
    - **Genetic Operators**: Batch-parallel implementations of Crossover (OX1), Swap, Relocate, 2-Opt*, and Swap*.
    - **Population Management**: `VectorizedPopulation` handles diversity maintenance and survivor selection on GPU.

Usage:
    Polices in this module generally adhere to a standard interface where they accept a problem instance
    (distance matrix, demands, capacity) and return a solution (routes and costs).
"""
