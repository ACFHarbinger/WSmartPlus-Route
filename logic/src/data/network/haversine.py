"""
Simple geometric distance strategies (Geodesic, Haversine, Euclidean).
"""

import math
from typing import Tuple

from logic.src.constants import EARTH_RADIUS

from .base import IterativeDistanceStrategy


class HaversineStrategy(IterativeDistanceStrategy):
    """Strategy for computing Haversine distances (Spherical)."""

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes distance using the Haversine formula (spherical Earth)."""
        coords_i_rad = (math.radians(coords_i[0]), math.radians(coords_i[1]))
        coords_j_rad = (math.radians(coords_j[0]), math.radians(coords_j[1]))
        dlat = coords_j_rad[0] - coords_i_rad[0]
        dlng = coords_j_rad[1] - coords_i_rad[1]
        a = math.sin(dlat / 2) ** 2 + math.cos(coords_i_rad[0]) * math.cos(coords_j_rad[0]) * math.sin(dlng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return c * EARTH_RADIUS
