"""
Pandas CSV dataset for simulation data stored in .csv files.

Expected CSV format (single sample):
    - Row 0: depot (fill values are 0)
    - Rows 1..N: bins
    - Columns: 'Lat'/'Lng' or 'Latitude'/'Longitude', and 'Day 0', 'Day 1', ..., 'Day D-1'
    - Optional: 'Max Fill' (per-bin max capacity)
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from logic.src.constants import MAX_CAPACITY_PERCENT

from .sim_dataset import SimulationDataset


class PandasCsvDataset(SimulationDataset):
    """
    Dataset wrapping simulation data loaded from a CSV file.

    The first row is the depot and the remaining rows are bins.
    Coordinates are automatically mapped from 'Lat'/'Lng' or 'Latitude'/'Longitude'.
    Daily fill values are expected in 'Day X' columns.
    """

    def __init__(self, sample: Dict[str, Any]):
        """Initialize the Pandas CSV dataset."""
        self._sample = sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return 1

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the sample at the given index."""
        if index != 0:
            raise IndexError("PandasCsvDataset only contains one sample.")
        return self._sample

    @staticmethod
    def load(path: str) -> "PandasCsvDataset":
        """Load a PandasCsvDataset from a .csv file."""
        df = pd.read_csv(path)
        sample = PandasCsvDataset._parse_df(df)
        return PandasCsvDataset(sample)

    @staticmethod
    def _parse_df(df: pd.DataFrame) -> Dict[str, Any]:
        """Parse a dataframe into a sample dict."""
        # Normalize coordinate column names
        if "Latitude" in df.columns and "Lat" not in df.columns:
            df = df.rename(columns={"Latitude": "Lat"})
        if "Longitude" in df.columns and "Lng" not in df.columns:
            df = df.rename(columns={"Longitude": "Lng"})

        depot = df.iloc[0]
        bins_df = df.iloc[1:]

        depot_coords = np.array([depot["Lat"], depot["Lng"]], dtype=np.float64)
        locs = bins_df[["Lat", "Lng"]].values.astype(np.float64)

        day_cols = sorted(
            [c for c in df.columns if isinstance(c, str) and c.startswith("Day ")],
            key=lambda c: int(c.split(" ", 1)[1]),
        )

        # waste shape: (n_days, n_bins)
        waste = bins_df[day_cols].values.astype(np.float64).T

        if "Max Fill" in df.columns:
            max_waste = bins_df["Max Fill"].values.astype(np.float64)
        else:
            max_waste = np.full(len(bins_df), MAX_CAPACITY_PERCENT, dtype=np.float64)

        if "ID" in df.columns:
            depot_id = df.iloc[0]["ID"]
            node_ids = bins_df["ID"].tolist()
        else:
            depot_id = 0
            node_ids = list(range(1, len(bins_df) + 1))

        return {
            "depot": depot_coords,
            "depot_id": depot_id,
            "locs": locs,
            "node_ids": node_ids,
            "waste": waste,
            "noisy_waste": waste,  # Noisy fallback to real waste if not provided
            "max_waste": max_waste,
        }
