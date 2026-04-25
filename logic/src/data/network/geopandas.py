"""
Strategy for computing distances using GeoPandas.

Attributes:
    GeoPandasStrategy: Strategy for computing distances using GeoPandas.

Example:
    >>> from logic.src.data.network.geopandas import GeoPandasStrategy
    >>> strategy = GeoPandasStrategy()
    >>> strategy.calculate(pd.DataFrame({"Lng": [0, 1], "Lat": [0, 1]}))
    array([[0.         , 111.31949079],
           [111.31949079 , 0.        ]])
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .base import DistanceStrategy


class GeoPandasStrategy(DistanceStrategy):
    """Strategy for computing distances using GeoPandas (Projected).

    Attributes:
        None
    """

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes distances using GeoPandas built-in distance functions.

        Args:
            coords: DataFrame with coordinates (must contain 'ID', 'lat', 'lng' columns).
            kwargs: Additional arguments for the distance strategy.

        Returns:
            Distance matrix as a NumPy array.
        """
        # World Geodetic System (https://epsg.io/4326)
        gdf = gpd.GeoDataFrame(
            coords,
            crs="EPSG:4326",
            geometry=gpd.points_from_xy(coords["Lng"], coords["Lat"]),
        )
        size = len(coords)
        distance_matrix = np.zeros((size, size))

        for id_row, row in gdf.iterrows():
            distance_matrix[id_row] = gdf["geometry"].distance(row["geometry"]) * 100

        return distance_matrix
