"""
Empirical sampling distributions.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Any, Optional, Tuple, cast

import numpy as np
import pandas
import torch

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.wsmart_bin_analysis import GridBase

from .base import BaseDistribution


class Empirical(BaseDistribution):
    """Sampling from an empirical dataset (e.g. file or Bins object)."""

    def __init__(
        self,
        grid: Optional[GridBase] = None,
        area: Optional[str] = None,
        indices: Optional[np.ndarray] = None,
        data_path: Optional[str] = None,
        dataset: Optional[Any] = None,
    ):
        """Initialize Class.

        Args:
            grid (Optional[GridBase]): Grid object.
            area (Optional[str]): Area name.
            data_path (Optional[str]): Path to data file/directory.
        """
        self.grid = grid
        self.dataset = dataset
        self.data_path = data_path
        if grid is None and data_path is not None and indices is not None and os.path.isdir(data_path):
            from logic.src.utils.data.loader import load_grid_base

            self.grid = load_grid_base(indices, area, data_path)

        if data_path is not None and os.path.isfile(data_path):
            if data_path.endswith(".pkl"):
                with open(data_path, "rb") as f:
                    self.dataset = pickle.load(f)
            elif data_path.endswith(".csv"):
                self.dataset = pandas.read_csv(data_path)
            elif data_path.endswith(".xlsx"):
                self.dataset = pandas.read_excel(data_path)
            elif data_path.endswith(".npz"):
                self.dataset = np.load(data_path)
            else:
                raise ValueError("Data path must be a directory or a pkl/csv/xlsx/npz file.")
        assert self.grid is not None or self.dataset is not None

    def _sample_tensor(self, size: Tuple[int, ...], generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.
            generator (Optional[torch.Generator], optional): Description of generator.

        Returns:
            torch.Tensor: Sampled values
        """
        if generator is None:
            generator = torch.Generator().manual_seed(42)
        batch_size = size[0]
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                # Use generator seed for pandas reproducibility
                rs = generator.initial_seed() if generator else 42
                vals = self.dataset.sample(n=batch_size, random_state=rs).values
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
            vals = self.grid.sample(n_samples=batch_size, rng=np.random.RandomState(generator.initial_seed()))
            vals_tensor = torch.from_numpy(vals).float() / 100.0
        else:
            raise ValueError("No grid or dataset found.")
        return torch.clip(vals_tensor, 0, 1)

    def _sample_array(self, size: Tuple[int, ...], rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.
            rng: Optional numpy RandomState for reproducibility.

        Returns:
            np.ndarray: Sampled values
        """
        if rng is None:
            rng = cast(np.random.RandomState, np.random)

        n_samples = size[0]
        if self.dataset is not None:
            if isinstance(self.dataset, pandas.DataFrame):
                sampled = self.dataset.sample(n=n_samples, random_state=rng).values
            elif isinstance(self.dataset, (np.ndarray, dict)):
                indices = rng.choice(len(self.dataset), n_samples)
                sampled = self.dataset[indices]
            elif isinstance(self.dataset, torch.Tensor):
                indices = rng.choice(len(self.dataset), n_samples)
                sampled = self.dataset[indices].cpu().numpy()
            else:
                raise ValueError("Dataset must be a pandas DataFrame, numpy array, dictionary, or torch Tensor.")
        elif self.grid is not None:
            sampled = self.grid.sample(n_samples=n_samples, rng=rng)
        else:
            raise ValueError("No grid or dataset found.")
        return np.clip(sampled, 0, 100)
