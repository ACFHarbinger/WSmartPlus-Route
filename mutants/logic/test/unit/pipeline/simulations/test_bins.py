"""Unit tests for the Bins class in logic/src/pipeline/simulations/bins.py."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from logic.src.pipeline.simulations.bins import Bins


class TestBinsStats:
    """Tests for statistical tracking (Welford's algorithm) in Bins."""

    def test_welford_algorithm(self, bins_stats):
        """
        Verify that Bins correctly maintains running mean and std dev
        using Welford's algorithm.
        """
        # Test sequence for 2 bins
        updates = [
            np.array([10, 5]),
            np.array([20, 5]),
            np.array([30, 5]),
            np.array([40, 5]),
            np.array([50, 5]),
        ]

        # Apply updates
        for filling in updates:
            bins_stats._process_filling(filling)

        # Check final statistics
        expected_means = np.array([30.0, 5.0])
        expected_std = np.array([np.std([10, 20, 30, 40, 50], ddof=1), 0.0])

        assert np.allclose(bins_stats.means, expected_means)
        assert np.allclose(bins_stats.std, expected_std)
        assert bins_stats.day_count == 5

    def test_set_statistics(self, bins_stats, tmp_path):
        """Test loading pre-computed statistics."""
        import pandas as pd
        stats_file = tmp_path / "stats.csv"
        df = pd.DataFrame({
            "Mean": [25.0, 10.0],
            "StD": [5.0, 2.0],
            "Count": [100, 100]
        })
        df.to_csv(stats_file, index=False)

        bins_stats.set_statistics("stats.csv")

        assert np.allclose(bins_stats.means, [25.0, 10.0])
        assert np.allclose(bins_stats.std, [5.0, 2.0])
        assert bins_stats.day_count == 100
        assert bins_stats.start_with_fill is True

        expected_sq_diff = np.array([25.0, 4.0]) * 99
        assert np.allclose(bins_stats.square_diff, expected_sq_diff)


class TestBinsFilling:
    """Tests for waste filling logic, verification of overflows and noise."""

    def test_fill_levels_update(self, mock_bins_params_loader, tmp_path):
        """Test simple filling update."""
        bins = Bins(n=2, data_dir=str(tmp_path))

        # Initial state 0
        filling = np.array([10.0, 20.0])

        # Deterministic fill
        overflows, fill_rates, levels, lost = bins._process_filling(filling)

        assert np.allclose(levels, [10.0, 20.0])
        assert overflows == 0
        assert lost == 0

    def test_overflow_and_loss(self, bins, mock_bins_params_loader):
        """
        Verify that waste exceeding 100% capacity is correctly tracked as 'lost'
        and capped at 100%.
        """
        bins.real_c = np.array([90.0] * 10) # 90% full

        n_overflows, fill, levels, lost = bins._process_filling(np.ones(10) * 20.0)

        assert n_overflows == 10
        assert np.allclose(bins.real_c, 100.0)
        # Lost Calculation: (10 / 100) * volume(1000) * density(10) = 0.1 * 1000 * 10 = 1000.0 kg per bin
        # Total Lost: 1000.0 * 10 bins = 10000.0 kg
        assert np.isclose(lost, 10000.0)
        assert np.allclose(bins.lost, 1000.0) # Cumulative per bin

    def test_noisy_filling(self, bins, mocker, mock_bins_params_loader):
        """Verify that noise is added to observed levels (c) but not real levels (real_c)."""
        # Mock numpy.random.normal to return fixed noise
        mocker.patch("numpy.random.normal", return_value=np.ones(10) * 5.0)

        bins.real_c = np.zeros(10)
        bins.c = np.zeros(10)

        # Add 10% waste
        fill = np.ones(10) * 10.0
        bins._process_filling(fill)

        # real_c should be 10.0
        assert np.allclose(bins.real_c, 10.0)

        # c should be 10.0 + 5.0 (noise) = 15.0
        assert np.allclose(bins.c, 15.0)


class TestBinsPrediction:
    """Tests for predictdaystooverflow logic."""

    def test_predict_days_math(self, tmp_path, mock_bins_params_loader):
        """Test the mathematical correctness of prediction."""
        # Use shared fixture to mock loader
        bins = Bins(n=1, data_dir=str(tmp_path), sample_dist="gamma", area="test", waste_type="test")

        # Case: Mean=10, Std=1 (Low variance), Capacity=0 (Empty)
        # Should take ~10 days to reach 100%
        ui = np.array([10.0])
        vi = np.array([1.0]) # Variance = std^2
        f = np.array([0.0])
        cl = 0.5 # median prediction

        days = bins._predictdaystooverflow(ui, vi, f, cl)

        assert 8 <= days[0] <= 12
