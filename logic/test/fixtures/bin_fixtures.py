"""
Fixtures for the Simulator pipeline - Bins fixtures.
"""

import numpy as np
import pytest
from logic.src.pipeline.simulations.bins import Bins

@pytest.fixture
def basic_bins(tmp_path):
    """Returns a basic Bins instance for testing."""
    return Bins(
        n=10,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="riomaior",
        waste_type="paper",
    )

@pytest.fixture
def mock_bins_instance(mocker):
    """Returns a high-level mock of a Bins instance."""
    mock_bins = mocker.MagicMock()
    mock_bins.inoverflow = np.array([10, 20])
    mock_bins.collected = np.array([100, 200])
    mock_bins.ncollections = np.array([1, 2])
    mock_bins.lost = np.array([5, 5])
    mock_bins.travel = 50.0
    mock_bins.ndays = 10
    mock_bins.n = 2
    mock_bins.profit = 100.0
    mock_bins.get_fill_history.return_value = np.array([[10, 20], [30, 40]])
    mock_bins.stochasticFilling.return_value = (
        0,
        np.zeros(2),
        np.zeros(2),
        0,
    )
    mock_bins.c = np.full(2, 50.0)  # 50% fill
    mock_bins.means = np.full(2, 10.0)
    mock_bins.std = np.full(2, 1.0)
    mock_bins.collectlevl = np.full(2, 90.0)  # For last_minute policy
    mock_bins.collect.return_value = (100.0, 2, 10.0, 5.0)

    return mock_bins

@pytest.fixture
def mock_bins_params_loader(mocker):
    """
    Mock config loader for Bins to avoid file system reads.
    Returns: vehicle_capacity, revenue, density, expenses, bin_volume
    """
    return mocker.patch(
        "logic.src.pipeline.simulations.bins.base.load_area_and_waste_type_params",
        return_value=(None, 1.0, 10.0, 0.5, 1000.0)
    )

@pytest.fixture
def bins_stats(tmp_path, mock_bins_params_loader):
    """Create a Bins instance for stats testing."""
    return Bins(
        n=2,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        area="test_area",
        waste_type="test_type",
    )

@pytest.fixture
def bins(tmp_path, mock_bins_params_loader):
    """Standard Bins fixture for general testing."""
    return Bins(
        n=10,
        data_dir=str(tmp_path),
        sample_dist="gamma",
        noise_mean=0.0,
        noise_variance=1.0  # Enable noise
    )
