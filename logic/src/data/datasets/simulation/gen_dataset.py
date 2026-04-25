"""
Generative simulation dataset that produces training examples on the fly.

This module provides a ``GenerativeDataset`` that lazily generates waste
samples using distribution classes from ``logic.src.data.distributions``,
without requiring pre-computed data files.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT

from .sim_dataset import SimulationDataset


class GenerativeDataset(SimulationDataset):
    """
    Dataset that generates waste simulation samples on the fly.

    Each ``__getitem__`` call produces a fresh sample dict with the same
    keys as other ``SimulationDataset`` subclasses:
        - ``'depot'``:       ``np.ndarray`` of shape ``(coord_dim,)``
        - ``'locs'``:        ``np.ndarray`` of shape ``(n_bins, coord_dim)``
        - ``'waste'``:       ``np.ndarray`` of shape ``(n_days, n_bins)``
        - ``'noisy_waste'``: ``np.ndarray`` of shape ``(n_days, n_bins)``
        - ``'max_waste'``:   ``np.float64`` scalar

    Samples are reproducible: each index gets a deterministic seed derived
    from ``base_seed + index``.

    Args:
        data_dir: Path to data directory.
        n_samples: Total number of virtual samples.
        n_days: Number of simulation days per sample.
        n_bins: Number of bins (problem size).
        distribution: Waste distribution name (e.g., ``'gamma1'``, ``'beta'``).
        depot: Depot coordinates ``(coord_dim,)`` or ``(1, coord_dim)``.
        locs: Bin locations ``(n_bins, coord_dim)`` or ``(1, n_bins, coord_dim)``.
        noise_mean: Mean of Gaussian sensor noise.
        noise_variance: Variance of Gaussian sensor noise.
        max_waste: Maximum waste capacity (percentage).
        grid: GridBase object for empirical sampling.
        seed: Random seed.
    """

    def __init__(
        self,
        data_dir: str,
        n_samples: int,
        n_days: int,
        n_bins: int,
        distribution: str = "gamma1",
        depot: Optional[np.ndarray] = None,
        locs: Optional[np.ndarray] = None,
        noise_mean: float = 0.0,
        noise_variance: float = 0.0,
        max_waste: float = MAX_CAPACITY_PERCENT,
        grid: Optional[Any] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the generative dataset."""
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.n_days = n_days
        self.n_bins = n_bins
        self.distribution = distribution
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.max_waste = max_waste
        self.grid = grid
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        # Default coordinates if not provided
        if depot is None:
            self.depot = np.array([0.5, 0.5])
        else:
            self.depot = np.asarray(depot).squeeze()

        if locs is None:
            self.locs = self.rng.uniform(size=(n_bins, 2))
        else:
            locs_arr = np.asarray(locs)
            self.locs = locs_arr.squeeze(0) if locs_arr.ndim == 3 else locs_arr

        self.waste_fills = self._generate_waste_days()
        self.noisy_waste_fills = self._apply_noise(self.waste_fills)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Generate a single simulation sample.

        Args:
            index: Sample index.

        Returns:
            Dict with keys: ``depot``, ``locs``, ``waste``, ``noisy_waste``, ``max_waste``.
        """
        return {
            "depot": self.depot.copy(),
            "locs": self.locs.copy(),
            "waste": self.waste_fills[index],
            "noisy_waste": self.noisy_waste_fills[index],
            "max_waste": np.asarray(self.max_waste),
        }

    def _generate_waste_days(self) -> np.ndarray:
        """Generate waste values for all simulation days.

        Returns:
            np.ndarray: Waste array of shape ``(n_days, n_bins)``.
        """
        from logic.src.data.generators.waste import generate_waste

        # Build graph tuple expected by generate_waste (batch_size=1 format)
        graph: Tuple[np.ndarray, np.ndarray] = (
            self.depot[None, :],  # (1, coord_dim)
            self.locs[None, :, :],  # (1, n_bins, coord_dim)
        )

        samples = []
        for _ in range(self.n_samples):
            waste_day = generate_waste(
                self.n_bins, self.distribution, graph, dataset_size=self.n_days, grid=self.grid, rng=self.rng
            )
            waste_day = np.clip(np.asarray(waste_day) * self.max_waste, 0, self.max_waste)
            samples.append(waste_day)

        return np.array(samples)  # (n_samples, n_days, n_bins)

    def _apply_noise(self, waste: np.ndarray) -> np.ndarray:
        """Apply Gaussian sensor noise to waste values.

        Args:
            waste: Clean waste array ``(n_days, n_bins)``.

        Returns:
            np.ndarray: Noisy waste array, same shape.
        """
        if self.noise_variance > 0:
            noise = self.rng.normal(self.noise_mean, np.sqrt(self.noise_variance), waste.shape)
            return np.clip(waste + noise, 0, self.max_waste)
        return waste.copy()
