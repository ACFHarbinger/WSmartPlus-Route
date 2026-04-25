"""
Euclidean Distance Strategy.

Attributes:
    EuclideanStrategy: Strategy for computing Euclidean distances.

Example:
    >>> from logic.src.data.network.euclidean import EuclideanStrategy
    >>> strategy = EuclideanStrategy()
    >>> strategy.calculate_pair((0, 0), (1, 1))
    21.054908551963157
"""

import math
from typing import Tuple

from .base import IterativeDistanceStrategy


class EuclideanStrategy(IterativeDistanceStrategy):
    """Strategy for computing Euclidean distances (Planar approximation).

    Attributes:
        None
    """

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes Euclidean distance with scaling for regional accuracy.

        Args:
            coords_i: Coordinates of the first point.
            coords_j: Coordinates of the second point.

        Returns:
            Euclidean distance between the two points.
        """
        dist = 86.51 * 1.58 * math.sqrt((coords_i[0] - coords_j[0]) ** 2 + (coords_i[1] - coords_j[1]) ** 2)
        return round(dist, 10)
