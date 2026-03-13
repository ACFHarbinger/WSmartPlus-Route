"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Rigorous implementation replacing "League Championship Algorithm (LCA)".
"""

from .params import MemeticAlgorithmToleranceBasedParams
from .solver import MemeticAlgorithmToleranceBasedSolver

__all__ = ["MemeticAlgorithmToleranceBasedSolver", "MemeticAlgorithmToleranceBasedParams"]
