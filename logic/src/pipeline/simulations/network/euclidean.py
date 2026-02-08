"""euclidean.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import euclidean
    """
import math
from typing import Tuple

from .base import IterativeDistanceStrategy


class EuclideanStrategy(IterativeDistanceStrategy):
    """Strategy for computing Euclidean distances (Planar approximation)."""

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes Euclidean distance with scaling for regional accuracy."""
        dist = 86.51 * 1.58 * math.sqrt((coords_i[0] - coords_j[0]) ** 2 + (coords_i[1] - coords_j[1]) ** 2)
        return round(dist, 10)
