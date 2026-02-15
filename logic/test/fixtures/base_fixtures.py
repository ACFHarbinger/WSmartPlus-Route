"""
Fixtures for the Simulator pipeline - Base fixtures.
"""

from multiprocessing import Lock, Value
import pytest

@pytest.fixture
def wsr_opts(tmp_path):
    """
    Provides a dictionary of default options (opts) for simulation tests.
    Tests can modify this fixture's output as needed.
    """
    # Create necessary subdirectories in tmp_path
    (tmp_path / "data" / "wsr_simulator").mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "assets" / "test_output" / "10_days" / "test_area_2"
    results_dir.mkdir(parents=True, exist_ok=True)

    return {
        "days": 10,
        "size": 2,
        "area": "riomaior",
        "waste_type": "paper",
        "policies": ["test_policy_gamma1"],
        "output_dir": "test_output",
        "checkpoint_dir": "temp",
        "resume": False,
        "distance_method": "hsd",
        "dm_filepath": None,
        "env_file": None,
        "gapik_file": None,
        "symkey_name": None,
        "edge_threshold": 50,
        "edge_method": "knn",
        "decoding.temperature": 1.0,
        "decoding.strategy": "greedy",
        "vertex_method": "mmn",
        "server_run": False,
        "gplic_file": None,
        "cache_regular": False,
        "waste_filepath": None,
        "n_vehicles": 1,
        "checkpoint_days": 5,
        "no_progress_bar": True,
        "data_distribution": "gamma",
        "n_samples": 1,
        "seed": 42,
        "problem": "vrpp",
        "stats_filepath": None,
        "model_path": None,
    }

@pytest.fixture
def mock_lock_counter(mocker):
    """
    Provides mock multiprocessing.Lock and Value objects.
    """
    mock_lock = mocker.MagicMock(spec=Lock())
    mock_counter = mocker.MagicMock(spec=Value("i"))
    mock_counter.value = 0
    # Mock the context manager for the lock
    mock_counter_lock = mocker.MagicMock()
    mock_counter_lock.__enter__.return_value = None
    mock_counter_lock.__exit__.return_value = None
    return mock_lock, mock_counter
