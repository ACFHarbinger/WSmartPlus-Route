"""Tests for distribution utilities."""

import torch
import pytest
from logic.src.data.distributions import (
    Cluster,
    Mixed,
    GaussianMixture,
    MixDistribution,
    MixMultiDistributions,
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
        dist = Cluster(n_cluster=3).set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mixed_output_range(self):
        """Verify Mixed data generation range."""
        batch = 4
        num_loc = 20
        dist = Mixed(n_cluster_mix=1).set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mix_distribution_coverage(self):
        """Verify MixDistribution runs without error."""
        batch = 10
        num_loc = 20
        dist = MixDistribution().set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)

    def test_gaussian_mixture(self):
        """Verify GaussianMixture generation."""
        batch = 4
        num_loc = 20
        # Mode 0 (Uniform)
        dist = GaussianMixture(num_modes=0).set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)

        # Mode > 1
        dist = GaussianMixture(num_modes=3, cdist=10).set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_mix_multi_distributions(self):
        """Verify MixMultiDistributions generation."""
        batch = 20  # Enough to likely hit multiple types
        num_loc = 20
        dist = MixMultiDistributions().set_sampling_method("sample_tensor")
        coords = dist.sample((batch, num_loc, 2))
        assert coords.shape == (batch, num_loc, 2)
        assert (coords >= 0).all() and (coords <= 1).all()

    def test_gamma_distribution(self):
        """Verify Gamma output shape."""
        batch, num_loc, dim = 4, 10, 2
        dist = Gamma(alpha=2.0, theta=1/2.0).set_sampling_method("sample_tensor")
        out = dist.sample((batch, num_loc, dim))
        assert out.shape == (batch, num_loc, dim)

    def test_empirical_with_data(self):
        """Verify Empirical with mock data."""
        batch, num_loc, dim = 2, 5, 2
        mock_data = torch.ones(10, 5, 2) * 0.5
        dist = Empirical(dataset=mock_data).set_sampling_method("sample_tensor")

        out = dist.sample((batch, 5, 2))
        assert out.shape == (batch, 5, 2)
        assert torch.allclose(out, torch.tensor(0.5))

    def test_empirical_with_bins(self):
        """Verify Empirical integration with Bins."""
        import numpy as np
        batch = 4
        num_loc = 10
        mock_grid = MagicMock()
        # load_filling returns np.ndarray in [0, 100]
        mock_grid.sample.return_value = np.random.uniform(0, 100, (batch, num_loc))

        dist = Empirical(grid=mock_grid).set_sampling_method("sample_tensor")
        out = dist.sample((batch, num_loc, 1))

        assert out.shape == (batch, num_loc)
        assert (out >= 0).all() and (out <= 1).all()
        # BaseDistribution.sample passes an rng to _sample_tensor, which Empirical passes to grid.sample
        mock_grid.sample.assert_called()
        call_kwargs = mock_grid.sample.call_args[1]
        assert call_kwargs["n_samples"] == batch
        assert "rng" in call_kwargs

    def test_gamma_vectorized(self):
        """Verify vectorized Gamma parameters."""
        batch, num_loc = 2, 5
        alpha = torch.ones(num_loc) * 2.0
        beta = torch.ones(num_loc) * 5.0
        dist = Gamma(alpha=alpha, theta=1/beta).set_sampling_method("sample_tensor")
        # size matches batch, num_loc
        out = dist.sample((batch,))
        assert out.shape == (batch, num_loc)
