"""Tests for distribution utilities."""

import torch
import pytest
from logic.src.data.distributions import (
    Cluster,
    Mixed,
    Gaussian_Mixture,
    Mix_Distribution,
    Mix_Multi_Distributions,
    Gamma,
    Empirical,
)
from unittest.mock import MagicMock


class TestDistributions:
    """Tests for distribution classes."""

    def test_cluster_output_shape(self):
        """Verify Cluster data generation shape."""
        batch = 4
        num_loc = 20
        dist = Cluster(n_cluster=3)
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mixed_output_range(self):
        """Verify Mixed data generation range."""
        batch = 4
        num_loc = 20
        dist = Mixed(n_cluster_mix=1)
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mix_distribution_coverage(self):
        """Verify Mix_Distribution runs without error."""
        batch = 10
        num_loc = 20
        dist = Mix_Distribution()
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)

    def test_gaussian_mixture(self):
        """Verify Gaussian_Mixture generation."""
        batch = 4
        num_loc = 20
        # Mode 0 (Uniform)
        dist = Gaussian_Mixture(num_modes=0)
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)

        # Mode > 1
        dist = Gaussian_Mixture(num_modes=3, cdist=10)
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mix_multi_distributions(self):
        """Verify Mix_Multi_Distributions generation."""
        batch = 20  # Enough to likely hit multiple types
        num_loc = 20
        dist = Mix_Multi_Distributions()
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_gamma_distribution(self):
        """Verify Gamma output shape."""
        batch, num_loc, dim = 4, 10, 2
        dist = Gamma(alpha=2.0, beta=2.0)
        out = dist.sample((batch, num_loc, dim))
        assert out.shape == (batch, num_loc, dim)

    def test_empirical_fallback(self):
        """Verify Empirical fallback to uniform."""
        batch, num_loc, dim = 4, 10, 2
        dist = Empirical(dataset_path=None)
        out = dist.sample((batch, num_loc, dim))
        assert out.shape == (batch, num_loc, dim)
        assert (out >= 0).all() and (out <= 1).all()

    def test_empirical_with_data(self):
        """Verify Empirical with mock data."""
        batch, num_loc, dim = 2, 5, 2
        mock_data = torch.ones(10, 5, 2) * 0.5
        dist = Empirical()
        dist.dataset = mock_data

        out = dist.sample((batch, 5, 2))
        assert out.shape == (batch, 5, 2)
        assert torch.allclose(out, torch.tensor(0.5))

    def test_empirical_with_bins(self):
        """Verify Empirical integration with Bins."""
        import numpy as np
        batch = 4
        num_loc = 10
        mock_bins = MagicMock()
        # stochasticFilling returns np.ndarray in [0, 100]
        mock_bins.stochasticFilling.return_value = np.random.uniform(0, 100, (batch, num_loc))

        dist = Empirical(bins=mock_bins)
        out = dist.sample((batch, num_loc, 1))

        assert out.shape == (batch, num_loc)
        assert (out >= 0).all() and (out <= 1).all()
        mock_bins.stochasticFilling.assert_called_with(n_samples=batch, only_fill=True)

    def test_gamma_vectorized(self):
        """Verify vectorized Gamma parameters."""
        batch, num_loc = 2, 5
        alpha = torch.ones(num_loc) * 2.0
        beta = torch.ones(num_loc) * 5.0
        dist = Gamma(alpha=alpha, beta=beta)
        # size matches batch, num_loc
        out = dist.sample((batch, num_loc))
        assert out.shape == (batch, num_loc)
