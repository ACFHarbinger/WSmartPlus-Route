import numpy as np
import pytest
from logic.src.pipeline.simulations.bins import Bins

class TestBins:
    """Class for Bins tests."""

    @pytest.mark.unit
    def test_bins_initialization(self, tmp_path, mock_bins_params_loader):
        """Test Bins initialization with default parameters."""
        # Use mocked params loader to avoid file system reads and NoneType error
        bins = Bins(n=10, data_dir=str(tmp_path), sample_dist="gamma", area="riomaior")
        assert bins.n == 10
        assert bins.ndays == 0
        assert len(bins.history) == 0
        assert bins.inoverflow.shape == (10,)
        assert bins.collected.shape == (10,)

    @pytest.mark.unit
    def test_collect_all(self, basic_bins):
        """Test collecting from all bins."""
        basic_bins.real_c = np.full(10, 50.0)
        basic_bins.c = np.full(10, 50.0)

        # ids should be 1-indexed for the depot-based tour logic
        ids = np.arange(0, 11) # [0, 1, 2, ..., 10]

        total_collected, sum_collected, ncollections, profit = basic_bins.collect(list(ids), cost=10.0)

        assert sum_collected > 0
        assert ncollections == 10
        assert np.all(basic_bins.real_c == 0)
        assert basic_bins.ndays == 1

    @pytest.mark.unit
    def test_collect_subset(self, basic_bins):
        """Test collecting from a subset of bins."""
        basic_bins.real_c = np.full(10, 50.0)
        basic_bins.c = np.full(10, 50.0)

        ids = [0, 1, 3, 5] # depot + bins 1, 3, 5
        total_collected, sum_collected, ncollections, profit = basic_bins.collect(ids, cost=5.0)

        assert ncollections == 3
        assert basic_bins.real_c[0] == 0  # Bin 1 (idx 0)
        assert basic_bins.real_c[1] == 50 # Bin 2 (idx 1)
        assert basic_bins.ndays == 1

    @pytest.mark.unit
    def test_stochastic_filling_basic(self, basic_bins):
        """Test basic stochastic filling logic."""
        basic_bins.setGammaDistribution(0) # Option 0 is valid
        inoverflow, filling, current_levels, sum_lost = basic_bins.stochasticFilling()

        assert len(filling) == 10
        assert len(current_levels) == 10
        assert len(basic_bins.history) == 1
        assert np.array_equal(basic_bins.history[-1], filling)

    @pytest.mark.unit
    def test_set_gamma_distribution(self, basic_bins):
        """Test setting gamma distribution parameters."""
        basic_bins.setGammaDistribution(1) # Option 1 is valid
        assert basic_bins.distribution == "gamma"
        assert len(basic_bins.dist_param1) == 10
        assert len(basic_bins.dist_param2) == 10
