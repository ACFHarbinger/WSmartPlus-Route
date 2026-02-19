"""
NumPy pickle dataset for legacy simulation data stored in .pkl files.
"""

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


class NumpyPickleDataset:
    """
    Dataset wrapping legacy pickle simulation data.

    The old format is a list of tuples per sample, where each tuple is either:
        - (depot, locs, waste, noisy_waste, max_waste) for SWCVRP
        - (depot, locs, waste, max_waste) for standard problems

    This class normalises access so that __getitem__ always returns a dict
    with the same keys as NumpyDictDataset.
    """

    def __init__(self, data: List[Tuple[Any, ...]]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        sample = self._data[index]
        if len(sample) == 5:
            depot, locs, waste, noisy_waste, max_waste = sample
        elif len(sample) == 4:
            depot, locs, waste, max_waste = sample
            noisy_waste = waste
        else:
            raise ValueError(f"Unexpected sample length: {len(sample)}")
        return {
            "depot": np.asarray(depot),
            "locs": np.asarray(locs),
            "waste": np.asarray(waste),
            "noisy_waste": np.asarray(noisy_waste),
            "max_waste": np.asarray(max_waste),
        }

    @staticmethod
    def load(path: str) -> "NumpyPickleDataset":
        """Load a NumpyPickleDataset from a .pkl file."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        return NumpyPickleDataset(data)
