"""
Simple geometric distance strategies (Geodesic, Haversine, Euclidean).
"""

import math
from typing import Tuple

from geopy.distance import geodesic

from logic.src.constants import EARTH_RADIUS

from .base import IterativeDistanceStrategy


class GeodesicStrategy(IterativeDistanceStrategy):
    """Strategy for computing geodesic distances (WGS84)."""

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes geodesic distance using the WGS84 ellipsoid."""
        return geodesic(coords_i, coords_j).km


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


class EuclideanStrategy(IterativeDistanceStrategy):
    """Strategy for computing Euclidean distances (Planar approximation)."""

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes Euclidean distance with scaling for regional accuracy."""
        dist = 86.51 * 1.58 * math.sqrt((coords_i[0] - coords_j[0]) ** 2 + (coords_i[1] - coords_j[1]) ** 2)
        return round(dist, 10)
