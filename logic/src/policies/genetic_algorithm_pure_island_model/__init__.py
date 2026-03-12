"""
Pure Island Model Genetic Algorithm for VRPP.

Rigorous implementation replacing "Soccer League Competition (SLC)".
Uses only genetic operators (crossover/mutation) without local search.
"""

from .params import PureIslandModelGAParams
from .solver import PureIslandModelGASolver

__all__ = ["PureIslandModelGASolver", "PureIslandModelGAParams"]
