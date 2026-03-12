"""
Genetic Algorithm with Stochastic Tournament Selection for VRPP.

Rigorous implementation replacing "League Championship Algorithm (LCA)".
"""

from .params import StochasticTournamentGAParams
from .solver import StochasticTournamentGASolver

__all__ = ["StochasticTournamentGASolver", "StochasticTournamentGAParams"]
