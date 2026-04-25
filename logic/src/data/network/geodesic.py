"""geodesic.py module.

Attributes:
    GeodesicStrategy: Strategy for computing geodesic distances on a sphere.

Example:
    >>> from logic.src.data.network.geodesic import GeodesicStrategy
    >>> strategy = GeodesicStrategy()
    >>> strategy.calculate_pair((0, 0), (1, 1))
    157.21777538020614
"""

from typing import Tuple

from geopy.distance import geodesic

from .base import IterativeDistanceStrategy


class GeodesicStrategy(IterativeDistanceStrategy):
    """Strategy for computing geodesic distances (WGS84).

    Attributes:
        None
    """

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes geodesic distance using the WGS84 ellipsoid.

        Args:
            coords_i: Coordinates of the first point.
            coords_j: Coordinates of the second point.

        Returns:
            Geodesic distance between the two points.
        """
        return geodesic(coords_i, coords_j).km
