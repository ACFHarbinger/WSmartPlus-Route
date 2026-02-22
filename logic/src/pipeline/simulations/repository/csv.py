"""
PandasCSV-based implementation of SimulationRepository.

Loads all simulation data (depot, coordinates) directly from a
PandasCsvDataset (.csv file) rather than from filesystem files.
"""

import numpy as np
import pandas as pd

from logic.src.data.datasets import PandasCsvDataset

from .base import SimulationRepository


class PandasCsvRepository(SimulationRepository):
    """
    Repository that sources simulation data from a PandasCsvDataset.

    The dataset is expected to contain at least depot, locs, and node_ids.
    Coordinates are stored in the dataset.
    """

    def __init__(self, dataset: PandasCsvDataset, sample_id: int = 0):
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
    ):
        """Builds data and bins_coordinates DataFrames from the dataset."""
        locs = self._sample["locs"]
        n_bins = locs.shape[0]
        # In CSV dataset, we might have more or fewer bins than requested if the file is fixed
        # but for now we follow PandasExcelRepository's assertion
        assert n_bins == number_of_bins, (
            f"Number of bins in dataset ({n_bins}) does not match number of bins requested ({number_of_bins})."
        )

        ids = self._sample["node_ids"]
        bins_coordinates = pd.DataFrame(
            {
                "ID": ids,
                "Lat": locs[:, 0],
                "Lng": locs[:, 1],
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
