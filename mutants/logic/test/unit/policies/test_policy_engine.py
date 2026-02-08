
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Import explicit modules for patch.object
import logic.src.policies.adapters.policy_bcp as policy_bcp_module
import logic.src.policies.adapters.policy_vrpp as policy_vrpp_module
import logic.src.policies.adapters.policy_hgs as policy_hgs_module
import logic.src.policies.adapters.policy_alns as policy_alns_module
import logic.src.policies.adapters.policy_sans as policy_sans_module

from logic.src.policies.adapters import PolicyRegistry, PolicyFactory
from logic.src.policies.adapters.policy_bcp import BCPPolicy
from logic.src.policies.adapters.policy_vrpp import VRPPPolicy
from logic.src.policies.adapters.policy_hgs import HGSPolicy
from logic.src.policies.adapters.policy_alns import ALNSPolicy
from logic.src.policies.adapters.policy_sans import SANSPolicy

class MockBins:
    def __init__(self, n=5):
        self.n = n
        self.c = np.ones(n) * 100.0  # Overflowing to trigger policy execution
        self.means = np.ones(n) * 1.0
        self.std = np.ones(n) * 0.1
        self.collectlevl = 90.0

@pytest.fixture
def mock_engine_data():
    n_bins = 5
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    # Create fake new_data (DataFrame)
    new_data = pd.DataFrame({
        "ID": range(n_bins + 1),
        "#bin": range(n_bins + 1),
        "Stock": [0.0] * (n_bins + 1),
        "Accum_Rate": [0.0] * (n_bins + 1)
    })

    # Create fake coords
    coords = pd.DataFrame({
        "ID": range(n_bins + 1),
        "Lat": [0.0] * (n_bins + 1),
        "Lng": [0.0] * (n_bins + 1)
    })

    return {
        "bins": MockBins(n_bins),
        "distance_matrix": dist_matrix,
        "distancesC": dist_matrix.astype(np.int32),
        "waste_type": "plastic",
        "area": "riomaior",
        "must_go": [1, 2, 3],  # Populate must_go to prevent early returns
        "new_data": new_data,
        "coords": coords,
        "config": {
            "bcp": {"bcp_engine": "ortools"},
            "hgs": {"engine": None},
            "alns": {"engine": None},
            "vrpp": {},
            "lookahead": {
                "sans": {},
                "Omega": 0.1
            }
        },
        "n_vehicles": 2,
        "model_env": MagicMock(),
        "run_tsp": False,
        "graph_size": n_bins
    }

def test_policy_factory_standardized():
    # Test that PolicyFactory correctly identifies policies from Registry or Fallback
    # Registry test (standard name)
    p = PolicyFactory.get_adapter("hgs")
    assert isinstance(p, HGSPolicy)

    # Fallback test (legacy name)
    p = PolicyFactory.get_adapter("policy_regular")
    from logic.src.policies.adapters.policy_tsp import TSPPolicy
    assert isinstance(p, TSPPolicy)

@pytest.mark.unit
def test_bcp_engine_override(mock_engine_data):
    with patch("logic.src.policies.adapters.policy_bcp.run_bcp") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0)

        policy = PolicyRegistry.get("bcp")()
        policy.execute(**mock_engine_data)

        # Verify run_bcp was called
        assert mock_run.called
        args, kwargs = mock_run.call_args
        # kwargs['env'] should be present
        assert "env" in kwargs

@pytest.mark.unit
def test_vrpp_engine_override(mocker, mock_engine_data):
    # VRPP Policy imports logic.src.utils.data.data_utils inside execute
    mocker.patch("logic.src.utils.data.data_utils.load_area_and_waste_type_params",
                 return_value=(4000, 0.16, 21.0, 1.0, 2.5))

    with patch("logic.src.policies.adapters.policy_vrpp.run_vrpp_optimizer") as mock_opt:
        mock_opt.return_value = ([0, 1, 0], 10.0)

        policy = PolicyRegistry.get("vrpp")()
        policy.execute(**mock_engine_data)

        assert mock_opt.called

@pytest.mark.unit
def test_hgs_engine_override(mock_engine_data):
    with patch("logic.src.policies.adapters.policy_hgs.run_hgs") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0, 5.0)

        policy = PolicyRegistry.get("hgs")()
        policy.execute(**mock_engine_data)

        assert mock_run.called

@pytest.mark.unit
def test_alns_engine_override(mock_engine_data):
    with patch("logic.src.policies.adapters.policy_alns.run_alns") as mock_run:
        mock_run.return_value = ([[1, 0]], 10.0, 5.0)

        # Set specific engine in config
        mock_engine_data["config"]["alns"]["engine"] = "ortools"

        policy = PolicyRegistry.get("alns")()
        policy.execute(**mock_engine_data)

        assert mock_run.called
        # Verify it passed the engine to run_alns
        args, _ = mock_run.call_args
        # run_alns(dist_matrix, demands, max_capacity, R, C, values, ...)
        # values is the 6th arg (index 5)
        values_arg = args[5]
        assert values_arg["engine"] == "ortools"

def test_sans_execution(mock_engine_data):
    # Patch improved_simulated_annealing in its home module (or where it's used)
    with patch("logic.src.policies.adapters.policy_sans.improved_simulated_annealing") as mock_sans, \
         patch("logic.src.utils.data.data_utils.load_area_and_waste_type_params") as mock_load:

        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)
        mock_sans.return_value = ([[1, 0]], 10.0, 5.0, 2.0, 10.0)

        policy = PolicyRegistry.get("sans")()
        policy.execute(**mock_engine_data)

        assert mock_sans.called
