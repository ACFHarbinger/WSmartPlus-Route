"""
Great Deluge (GD) acceptance criterion.
"""

from .policy_gd import GreatDelugePolicy
from .solver import GDSolver

__all__ = ["GreatDelugePolicy", "GDSolver"]
