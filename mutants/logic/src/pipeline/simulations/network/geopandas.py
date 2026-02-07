"""
Strategy for computing distances using GeoPandas.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .base import DistanceStrategy


class GeoPandasStrategy(DistanceStrategy):
    """Strategy for computing distances using GeoPandas (Projected)."""

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes distances using GeoPandas built-in distance functions.
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
