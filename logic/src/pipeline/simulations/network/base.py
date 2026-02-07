"""
Abstract base classes for distance matrix calculation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class DistanceStrategy(ABC):
    """
    Abstract base class for distance matrix calculation strategies.

    Defines the interface for computing pairwise distances between
    geographic coordinates. Each strategy implements a different
    calculation method (API-based, formula-based, or network-based).
    """

    @abstractmethod
    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Computes pairwise distance matrix for given coordinates.

        Args:
            coords: DataFrame with 'Lat' and 'Lng' columns
            **kwargs: Strategy-specific parameters

        Returns:
            np.ndarray: n×n symmetric distance matrix in kilometers
        """
        pass

    def _eval_kwarg(self, kwarg: str, kwargs: Dict[str, Any]) -> bool:
        """Check if keyword argument exists and is not None."""
        return True if kwarg in kwargs and kwargs[kwarg] is not None else False


class IterativeDistanceStrategy(DistanceStrategy):
    """Base class for strategies that compute distances pairwise iteratively."""

    @abstractmethod
    def calculate_pair(self, coords_i: Tuple[float, float], coords_j: Tuple[float, float]) -> float:
        """
        Computes distance between a single pair of coordinates.
        """
        pass

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Iterates through all pairs to compute a full distance matrix.
        """
        size = len(coords)
        distance_matrix = np.zeros((size, size))

        verbose = kwargs.get("verbose", False)
        rows = list(coords.iterrows())
        for id_i, row_i in tqdm(rows, total=size, desc="Outer Loop", disable=not verbose):
            coords_i = (row_i["Lat"], row_i["Lng"])
            for id_j, row_j in tqdm(rows, total=size, desc="Inner Loop", leave=False, disable=not verbose):
                if id_i != id_j:
                    coords_j = (row_j["Lat"], row_j["Lng"])
                    distance_matrix[id_i, id_j] = self.calculate_pair(coords_i, coords_j)

        return distance_matrix
