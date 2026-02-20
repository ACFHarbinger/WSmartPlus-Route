"""
NumpyDict-based implementation of SimulationRepository.

Loads all simulation data (depot, coordinates, waste) directly from a
NumpyDictDataset (.npz file) rather than from filesystem CSV/Excel files.
Indices are simply 0..N-1 since the dataset already contains only the
relevant bins.
"""

import numpy as np
import pandas as pd

from logic.src.data.datasets import NumpyDictDataset

from .base import SimulationRepository


class NumpyDictRepository(SimulationRepository):
    """
    Repository that sources simulation data from a NumpyDictDataset.

    The dataset is expected to contain at least:
        - 'depot': (n_samples, coord_dim)
        - 'locs':  (n_samples, n_nodes, coord_dim)

    Since the dataset already holds only the selected bins, indices are
    trivially 0..N-1 and no filesystem lookup is needed.
    """

    def __init__(self, dataset: NumpyDictDataset, sample_id: int = 0):
        self._dataset = dataset
        self._sample = dataset[sample_id]

    def set_sample(self, sample_id: int) -> None:
        """Switch to a different sample within the dataset."""
        self._sample = self._dataset[sample_id]

    def get_indices(self, filename, n_samples, n_nodes, data_size, lock=None):
        """Returns trivial indices [0..n_nodes-1] for each sample."""
        indices = list(range(n_nodes))
        return [indices] * n_samples

    def get_depot(self, area=None, data_dir=None):
        """Builds a depot DataFrame from the dataset's depot array."""
        depot = self._sample["depot"]
        return pd.DataFrame(
            {
                "ID": [0],
                "Lat": [float(depot[1])],
                "Lng": [float(depot[0])],
                "Stock": [0.0],
                "Accum_Rate": [0.0],
            }
        )

    def get_simulator_data(
        self,
        number_of_bins,
        area="Rio Maior",
        waste_type=None,
        lock=None,
        data_dir=None,
    ):
        """Builds data and bins_coordinates DataFrames from the dataset."""
        locs = self._sample["locs"]
        n_bins = locs.shape[0]
        assert n_bins == number_of_bins, "Number of bins in dataset does not match number of bins requested."

        ids = self._sample["node_ids"][1:]
        bins_coordinates = pd.DataFrame(
            {
                "ID": ids,
                "Lat": locs[:, 1],
                "Lng": locs[:, 0],
            }
        )
        data = pd.DataFrame(
            {
                "ID": ids,
                "Stock": np.zeros(n_bins),
                "Accum_Rate": np.zeros(n_bins),
            }
        )
        return data, bins_coordinates

    def get_area_params(self, area, waste_type):
        """Delegates to data_utils for area/waste parameters."""
        from logic.src.utils.data.data_utils import load_area_and_waste_type_params

        return load_area_and_waste_type_params(area, waste_type)
