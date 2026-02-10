"""
Distance Calculation and Network Sparsification.
"""

import os
from typing import Any, Union

import numpy as np
import pandas as pd

from logic.src.constants import EARTH_RADIUS, ROOT_DIR

from .base import DistanceStrategy, IterativeDistanceStrategy
from .euclidean import EuclideanStrategy
from .file import FileStrategy
from .geodesic import GeodesicStrategy
from .geopandas import GeoPandasStrategy
from .google import GoogleMapsStrategy
from .haversine import HaversineStrategy
from .osm import OSMStrategy


def haversine_distance(
    lat1: Union[float, np.ndarray, pd.Series],
    lng1: Union[float, np.ndarray, pd.Series],
    lat2: Union[float, np.ndarray, pd.Series],
    lng2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Stand-alone vectorized Haversine distance calculation.
    """
    lat1, lng1, lat2, lng2 = np.radians(lat1), np.radians(lng1), np.radians(lat2), np.radians(lng2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS * c


def compute_distance_matrix(coords: pd.DataFrame, method: str, **kwargs: Any) -> np.ndarray:
    """
    Computes or loads cached pairwise distance matrix for bin locations.
    """
    STRATEGIES = {
        "gmaps": GoogleMapsStrategy,
        "gpd": GeoPandasStrategy,
        "osm": OSMStrategy,
        "gdsc": GeodesicStrategy,
        "hsd": HaversineStrategy,
        "ogd": EuclideanStrategy,
        "file": FileStrategy,
    }

    assert method in STRATEGIES, f"Method {method} not supported. usage: {list(STRATEGIES.keys())}"

    # Caching Logic
    to_save = False
    matrix_path = None

    if "dm_filepath" in kwargs and kwargs["dm_filepath"] is not None:
        dm_filepath = kwargs["dm_filepath"]
        filename_only = os.path.basename(dm_filepath) == dm_filepath and not os.path.isabs(dm_filepath)
        matrix_path = (
            os.path.join(
                ROOT_DIR,
                "data",
                "wsr_simulator",
                "distance_matrix",
                dm_filepath,
            )
            if filename_only
            else dm_filepath
        )

        if os.path.isfile(matrix_path):
            # Use FileStrategy for loading
            strategy = FileStrategy()
            return strategy.calculate(coords, **kwargs)
        else:
            # Prepare for saving
            os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
            with open(matrix_path, mode="w", newline="") as matrix_f:
                matrix_f.write(",".join(map(str, coords["ID"].to_numpy())) + "\n")
            to_save = True

    # Strategy Execution
    strategy_cls = STRATEGIES[method]
    strategy = strategy_cls()  # type: ignore[abstract, assignment]

    kwargs["verbose"] = to_save or kwargs.get("verbose", False)

    distance_matrix = strategy.calculate(coords, **kwargs)

    # Saving Result
    if to_save:
        with open(matrix_path, mode="a", newline="") as matrix_f:  # type: ignore[arg-type]
            for row in distance_matrix:
                matrix_f.write(",".join(map(str, row)) + "\n")

    return distance_matrix


__all__ = [
    "DistanceStrategy",
    "IterativeDistanceStrategy",
    "GoogleMapsStrategy",
    "GeoPandasStrategy",
    "OSMStrategy",
    "GeodesicStrategy",
    "HaversineStrategy",
    "EuclideanStrategy",
    "FileStrategy",
    "haversine_distance",
    "compute_distance_matrix",
]
