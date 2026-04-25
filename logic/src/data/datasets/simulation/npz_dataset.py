"""
NumPy dict dataset for simulation data stored in .npz files.
"""

from typing import Dict

import numpy as np

from .sim_dataset import SimulationDataset


class NumpyDictDataset(SimulationDataset):
    """
    Dataset wrapping a dict of named numpy arrays loaded from a .npz file.

    Expected structure (from VRPInstanceBuilder.build()):
        - 'depot': (n_samples, coord_dim)
        - 'locs': (n_samples, n_nodes, coord_dim)
        - 'waste': (n_samples, n_days, n_nodes)
        - 'noisy_waste': (n_samples, n_days, n_nodes)
        - 'max_waste': (n_samples,)
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        """Initialize the NumPy dict dataset."""
        self.data = data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        first_key = next(iter(self.data))
        return self.data[first_key].shape[0]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Return the sample at the given index."""
        return {key: arr[index] for key, arr in self.data.items()}

    @staticmethod
    def load(path: str) -> "NumpyDictDataset":
        """Load a NumpyDictDataset from a .npz file."""
        return NumpyDictDataset(dict(np.load(path)))

    def save(self, path: str) -> None:
        """Save the dataset to a .npz file."""
        np.savez_compressed(path, **self.data)
