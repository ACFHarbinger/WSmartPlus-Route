"""
Shared pytest fixtures for the wsmart_bin_analysis test suite.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ..Deliverables import simulation
from ..Deliverables.container import Container


@pytest.fixture
def sample_df_fill():
    """Fixture providing a sample dataframe of bin fill levels."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="H")
    data = {"Date": dates, "ID": [1] * 100, "Fill": np.random.randint(0, 100, size=100)}
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_collection():
    """Fixture providing a sample dataframe of collection records."""
    dates = pd.date_range(start="2020-01-01", periods=5, freq="24H")
    data = {"Date": dates, "ID": [1] * 5}
    return pd.DataFrame(data)


@pytest.fixture
def sample_info():
    """Fixture providing sample container metadata."""
    data = {"ID": [1], "Freguesia": ["TestLoc"], "Latitude": [0.0], "Longitude": [0.0]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_container(sample_df_fill, sample_df_collection, sample_info):
    """Fixture providing a Container instance pre-loaded with sample data."""
    return Container(sample_df_fill, sample_df_collection, sample_info)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture providing a temporary directory structured for data persistence tests."""
    # Create structure for simulation/save_load tests
    d = tmp_path / "data"
    d.mkdir()
    (d / "bins_waste").mkdir()
    (d / "coordinates").mkdir()
    return str(d)


@pytest.fixture
def mock_subprocess():
    """Fixture mocking subprocess.run for testing external command execution (like R scripts)."""
    with patch("subprocess.run") as mock:
        yield mock


@pytest.fixture
def predictor_data():
    """Fixture providing training and testing datasets for SARIMA predictor tests."""
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    train = pd.DataFrame(np.random.rand(40, 5), index=dates[:40], columns=[f"Bin_{i}" for i in range(5)])
    test = pd.DataFrame(np.random.rand(10, 5), index=dates[40:], columns=[f"Bin_{i}" for i in range(5)])
    return train, test, dates


@pytest.fixture
def mock_load_data():
    """Fixture mocking the simulation data loading functions."""
    with (
        patch.object(simulation, "load_rate_series") as mock_rate,
        patch.object(simulation, "load_info") as mock_info,
        patch.object(simulation, "load_rate_global_wrapper") as mock_wrapper,
    ):
        yield mock_rate, mock_info, mock_wrapper
