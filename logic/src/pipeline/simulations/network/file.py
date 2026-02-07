"""
Strategy for loading distance matrices from files.
"""

import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

from logic.src.constants import ROOT_DIR

from .base import DistanceStrategy


class FileStrategy(DistanceStrategy):
    """Strategy for loading distance matrices from disk."""

    def calculate(self, coords: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """
        Loads a pre-computed distance matrix from a CSV file.

        Args:
            coords: DataFrame with bin coordinates (used for validation if needed).
            **kwargs: Must include 'dm_filepath'.

        Returns:
            np.ndarray: Loaded distance matrix.
        """
        assert self._eval_kwarg("dm_filepath", kwargs), "Missing 'dm_filepath' in kwargs for FileStrategy."
        dm_filepath = kwargs["dm_filepath"]

        # Path resolution (consistent with compute_distance_matrix)
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

        if not os.path.isfile(matrix_path):
            raise FileNotFoundError(f"Distance matrix file not found: {matrix_path}")

        # Load matrix, skipping first row and column (IDs)
        distance_matrix = np.loadtxt(matrix_path, delimiter=",")[1:, 1:]

        # Handle focus_idx if present
        if self._eval_kwarg("focus_idx", kwargs):
            focus_idx = kwargs["focus_idx"]
            idx_list = list(focus_idx[0]) if isinstance(focus_idx[0], Iterable) else list(focus_idx)
            # Add 0 for depot (assumed to be at index 0 in the original matrix)
            # The matrix loaded [1:, 1:] already maps to indices [0...N-1] where 0 is depot
            idx = np.array([-1] + idx_list) + 1
            return distance_matrix[idx[:, None], idx]

        return distance_matrix
