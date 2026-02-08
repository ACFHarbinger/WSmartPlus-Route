"""geodesic.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import geodesic
    """
from typing import Tuple

from geopy.distance import geodesic

from .base import IterativeDistanceStrategy


class GeodesicStrategy(IterativeDistanceStrategy):
    """Strategy for computing geodesic distances (WGS84)."""

    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """Computes geodesic distance using the WGS84 ellipsoid."""
        return geodesic(coords_i, coords_j).km
