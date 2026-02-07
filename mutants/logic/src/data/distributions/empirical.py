"""
Empirical sampling distributions.
"""

from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.bins import Bins


class Empirical:
    """Sampling from an empirical dataset (e.g. file or Bins object)."""

    def __init__(self, bins: Optional[Bins] = None, dataset_path: Optional[str] = None):
        self.bins = bins
        self.dataset = None
        if bins is None and dataset_path is not None and os.path.exists(dataset_path):
            with open(dataset_path, "rb") as f:
                self.dataset = pickle.load(f)

            if not isinstance(self.dataset, torch.Tensor):
                try:
                    if isinstance(self.dataset, np.ndarray):
                        self.dataset = torch.from_numpy(self.dataset)
                    elif isinstance(self.dataset, list):
                        self.dataset = torch.tensor(self.dataset)
                except Exception:
                    pass

    def sample(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Sample from empirical dataset.

        Args:
            size: Sampling shape. First dimension is assumed to be batch size.
        """
        batch_size = size[0]

        if self.bins is not None:
            # stochasticFilling(only_fill=True) returns np.ndarray
            vals = self.bins.stochasticFilling(n_samples=batch_size, only_fill=True)
            if isinstance(vals, np.ndarray):
                vals_tensor = torch.from_numpy(vals).float()
                return vals_tensor / 100.0
            return torch.rand(*size)

        if self.dataset is None:
            return torch.rand(*size)

        if isinstance(self.dataset, torch.Tensor):
            indices = torch.randint(0, len(self.dataset), (batch_size,))
            return self.dataset[indices]

        return torch.rand(*size)
