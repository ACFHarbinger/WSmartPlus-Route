"""
Unified dataset-backed implementation of SimulationRepository.

Loads all simulation data (depot, coordinates, waste) from any supported
dataset type (NumpyDictDataset, PandasExcelDataset, PandasCsvDataset)
rather than from filesystem CSV/Excel files.

Attributes:
    DatasetRepository: Repository sourcing data from WSmart-style datasets.

Example:
    >>> # dataset = NumpyDictDataset.load("data.pkl")
    >>> # repo = DatasetRepository(dataset)
"""

from typing import Any, List, Optional, Tuple, Union

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

    Attributes:
        dataset: The source dataset object.
        sample_id: Index of the currently active sample.
    """

    def __init__(self, dataset: _DatasetType, sample_id: int = 0):
        """
        Initializes the dataset repository.

        Args:
            dataset: The dataset object containing simulation samples.
            sample_id: Index of the initial sample to use.
        """
        self._dataset = dataset
        self._sample = dataset[sample_id]

    def set_sample(self, sample_id: int) -> None:
        """Switch to a different sample within the dataset.

        Args:
            sample_id: Index of the sample to activate.
        """
        self._sample = self._dataset[sample_id]

    def get_indices(
        self, filename: Any, n_samples: int, n_nodes: int, data_size: int, lock: Optional[Any] = None
    ) -> List[List[int]]:
        """Returns node indices from the dataset, or trivial [0..n_nodes-1].

        Args:
            filename: Path to the index file (unused in this implementation).
            n_samples: Number of samples to generate/load.
            n_nodes: Number of nodes per sample.
            data_size: Total size of the source data.
            lock: Optional file lock.

        Returns:
            A list containing the same set of indices for each requested sample.
        """
        ids = list(self._sample["node_ids"]) if "node_ids" in self._sample else list(range(n_nodes))
        return [ids] * n_samples

    def get_depot(self, area: Any, data_dir: Optional[str] = None) -> pd.DataFrame:
        """Builds a depot DataFrame from the dataset's depot array.

        Args:
            area: Name of the area (unused).
            data_dir: Path to the data directory (unused).

        Returns:
            Pandas DataFrame containing depot coordinates.
        """
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
        number_of_bins: int,
        area: str = "Rio Maior",
        waste_type: Optional[str] = None,
        lock: Optional[Any] = None,
        data_dir: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Builds data and bins_coordinates DataFrames from the dataset.

        Args:
            number_of_bins: Number of bins to load.
            area: Name of the area (unused).
            waste_type: Type of waste (unused).
            lock: Optional file lock.
            data_dir: Path to the data directory (unused).

        Returns:
            Tuple containing (stats_df, coordinates_df).
        """
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
