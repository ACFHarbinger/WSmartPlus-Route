"""
Generative simulation dataset that produces training examples on the fly.

This module provides a ``GenerativeDataset`` that lazily generates waste
samples using distribution classes from ``logic.src.data.distributions``,
without requiring pre-computed data files.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from logic.src.constants.routing import MAX_CAPACITY_PERCENT
from logic.src.data.generators.waste import generate_waste

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
        base_seed: Base random seed for reproducibility.
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
        base_seed: int = 0,
    ):
        self.data_dir = data_dir
        self.n_samples = n_samples
        self.n_days = n_days
        self.n_bins = n_bins
        self.distribution = distribution
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.max_waste = max_waste
        self.base_seed = base_seed

        # Default coordinates if not provided
        if depot is None:
            self.depot = np.array([0.5, 0.5])
        else:
            self.depot = np.asarray(depot).squeeze()

        if locs is None:
            rng = np.random.RandomState(base_seed)
            self.locs = rng.uniform(size=(n_bins, 2))
        else:
            locs_arr = np.asarray(locs)
            self.locs = locs_arr.squeeze(0) if locs_arr.ndim == 3 else locs_arr

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Generate a single simulation sample.

        Args:
            index: Sample index (used with ``base_seed`` for reproducibility).

        Returns:
            Dict with keys: ``depot``, ``locs``, ``waste``, ``noisy_waste``, ``max_waste``.
        """
        # Seed for reproducibility per sample
        rng_state = np.random.get_state()
        np.random.seed(self.base_seed + index)
        try:
            waste = self._generate_waste_days()
            noisy_waste = self._apply_noise(waste)
        finally:
            np.random.set_state(rng_state)

        return {
            "depot": self.depot.copy(),
            "locs": self.locs.copy(),
            "waste": waste,
            "noisy_waste": noisy_waste,
            "max_waste": np.asarray(self.max_waste),
        }

    def _generate_waste_days(self) -> np.ndarray:
        """Generate waste values for all simulation days.

        Returns:
            np.ndarray: Waste array of shape ``(n_days, n_bins)``.
        """
        # Build graph tuple expected by generate_waste (batch_size=1 format)
        graph: Tuple[np.ndarray, np.ndarray] = (
            self.depot[None, :],  # (1, coord_dim)
            self.locs[None, :, :],  # (1, n_bins, coord_dim)
        )

        days = []
        for _ in range(self.n_days):
            # generate_waste with dataset_size=1 returns (n_bins,)
            waste_day = generate_waste(self.n_bins, self.distribution, graph, dataset_size=1)
            waste_day = np.clip(np.asarray(waste_day) * self.max_waste, 0, self.max_waste)
            days.append(waste_day)

        return np.array(days)  # (n_days, n_bins)

    def _apply_noise(self, waste: np.ndarray) -> np.ndarray:
        """Apply Gaussian sensor noise to waste values.

        Args:
            waste: Clean waste array ``(n_days, n_bins)``.

        Returns:
            np.ndarray: Noisy waste array, same shape.
        """
        if self.noise_variance > 0:
            noise = np.random.normal(self.noise_mean, np.sqrt(self.noise_variance), waste.shape)
            return np.clip(waste + noise, 0, self.max_waste)
        return waste.copy()
