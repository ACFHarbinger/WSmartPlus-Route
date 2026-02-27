"""
Unified dataset-backed implementation of SimulationRepository.

Loads all simulation data (depot, coordinates, waste) from any supported
dataset type (NumpyDictDataset, PandasExcelDataset, PandasCsvDataset)
rather than from filesystem CSV/Excel files.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

from logic.src.data.datasets import NumpyDictDataset, PandasCsvDataset, PandasExcelDataset

from .base import SimulationRepository

# Supported dataset types
_DatasetType = Union[NumpyDictDataset, PandasExcelDataset, PandasCsvDataset]


class DatasetRepository(SimulationRepository):
    """
    Repository that sources simulation data from any supported dataset.

    All dataset formats are expected to store coordinates in ``(lat, lng)``
    order and include ``node_ids`` as bin-only (no depot).
    """

    def __init__(self, dataset: _DatasetType, sample_id: int = 0):
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
        depot_id = self._sample.get("depot_id", 0)

        return pd.DataFrame(
            {
                "ID": [depot_id],
                "Lat": [float(depot[0])],
                "Lng": [float(depot[1])],
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Builds data and bins_coordinates DataFrames from the dataset."""
        locs = self._sample["locs"]
        n_bins = locs.shape[0]
        assert n_bins == number_of_bins, (
            f"Number of bins in dataset ({n_bins}) does not match number requested ({number_of_bins})."
        )

        ids = self._sample["node_ids"]

        bins_coordinates = pd.DataFrame({"ID": ids, "Lat": locs[:, 0], "Lng": locs[:, 1]})
        data = pd.DataFrame(
            {
                "ID": ids,
                "Stock": np.zeros(n_bins),
                "Accum_Rate": np.zeros(n_bins),
            }
        )
        return data, bins_coordinates
