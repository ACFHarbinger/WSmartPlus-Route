"""
Empirical sampling distributions.
"""

from __future__ import annotations

import os
import pickle
import pandas
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from logic.src.utils.data.loader import load_grid_base
    from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase


class Empirical:
    """Sampling from an empirical dataset (e.g. file or Bins object)."""

    def __init__(self, grid: Optional[GridBase] = None, area: Optional[str] = None, indices: Optional[np.ndarray] = None, data_path: Optional[str] = None):
        """Initialize Class.

        Args:
            grid (Optional[GridBase]): Grid object.
            area (Optional[str]): Area name.
            data_path (Optional[str]): Path to data file/directory.
        """
        self.grid = grid
        self.dataset = None
        self.data_path = data_path
        if grid is None and data_path is not None and indices is not None and os.path.isdir(data_path):
            self.grid = load_grid_base(data_path, indices, area)

        if data_path is not None and os.path.isfile(data_path):
            if data_path.endswith(".pkl"):
                self.dataset = pickle.load(open(data_path, "rb"))
            elif data_path.endswith(".csv"):
                self.dataset = pandas.read_csv(data_path)
            elif data_path.endswith(".xlsx"):
                self.dataset = pandas.read_excel(data_path)
            elif data_path.endswith(".npz"):
                self.dataset = np.load(data_path)
            else:
                raise ValueError("Data path must be a directory or a pkl/csv/xlsx/npz file.")
        assert self.grid is not None or self.dataset is not None

    def sample_tensor(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.

        Returns:
            torch.Tensor: Sampled values
        """
        batch_size = size[0]
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                vals = self.dataset.sample(n=batch_size).values
                vals_tensor = torch.tensor(vals).float() / 100.0
            elif isinstance(self.dataset, np.ndarray):
                vals = self.dataset[np.random.choice(len(self.dataset), batch_size)]
                vals_tensor = torch.from_numpy(vals).float() / 100.0
            elif isinstance(self.dataset, dict):
                vals = self.dataset[np.random.choice(len(self.dataset), batch_size)]
                vals_tensor = torch.tensor(vals).float() / 100.0
            elif isinstance(self.dataset, torch.Tensor):
                vals_tensor = self.dataset[np.random.choice(len(self.dataset), batch_size)]
            else:
                raise ValueError("Dataset must be a pandas DataFrame, numpy array, dictionary, or torch Tensor.")
        elif self.grid is not None:
            vals = self.grid.sample(n_samples=batch_size)
            vals_tensor = torch.from_numpy(vals).float() / 100.0
        else:
            raise ValueError("No grid or dataset found.")
        return vals_tensor

    def sample_array(self, size: Tuple[int, ...]) -> np.ndarray:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.

        Returns:
            np.ndarray: Sampled values
        """
        n_samples = size[0]
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                return self.dataset.sample(n=n_samples).values
            elif isinstance(self.dataset, np.ndarray):
                return self.dataset[np.random.choice(len(self.dataset), n_samples)]
            elif isinstance(self.dataset, dict):
                return self.dataset[np.random.choice(len(self.dataset), n_samples)]
            elif isinstance(self.dataset, torch.Tensor):
                return self.dataset[np.random.choice(len(self.dataset), n_samples)].cpu().numpy()
            else:
                raise ValueError("Dataset must be a pandas DataFrame, numpy array, dictionary, or torch Tensor.")
        elif self.grid is not None:
            return self.grid.sample(n_samples=n_samples)
        else:
            raise ValueError("No grid or dataset found.")
