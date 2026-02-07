"""Tests for distribution utilities."""

import torch
import pytest
from logic.src.data.distributions import (
    Cluster,
    Mixed,
    Gaussian_Mixture,
    Mix_Distribution,
    Mix_Multi_Distributions,
)


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
