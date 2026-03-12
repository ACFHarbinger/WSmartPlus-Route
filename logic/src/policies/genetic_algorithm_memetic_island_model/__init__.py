"""
Memetic Island Model Genetic Algorithm for VRPP.

Rigorous implementation replacing "Hybrid Volleyball Premier League (HVPL)".
Combines island model topology with local search (ALNS) on every individual.
"""

from .params import MemeticIslandModelGAParams
from .solver import MemeticIslandModelGASolver

__all__ = ["MemeticIslandModelGASolver", "MemeticIslandModelGAParams"]
