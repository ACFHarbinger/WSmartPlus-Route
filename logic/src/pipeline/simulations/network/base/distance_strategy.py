"""distance_strategy.py module.

    Attributes:
        MODULE_VAR (Type): Description of module level variable.

    Example:
        >>> import distance_strategy
    """
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


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
