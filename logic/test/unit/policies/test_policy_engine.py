
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# Import explicit modules for patch.object
import logic.src.policies.policy_bcp as policy_bcp_module
import logic.src.policies.policy_vrpp as policy_vrpp_module
import logic.src.policies.policy_hgs as policy_hgs_module
import logic.src.policies.policy_alns as policy_alns_module
import logic.src.policies.policy_sans as policy_sans_module
import logic.src.policies.policy_lac as policy_lac_module

from logic.src.policies.adapters import PolicyRegistry
from logic.src.policies.policy_bcp import BCPPolicy
from logic.src.policies.policy_vrpp import VRPPPolicy
from logic.src.policies.policy_hgs import HGSPolicy
from logic.src.policies.policy_alns import ALNSPolicy
from logic.src.policies.policy_sans import SANSPolicy
from logic.src.policies.policy_lac import LACPolicy

class MockBins:
    def __init__(self, n=5):
        self.c = np.ones(n) * 100.0  # Overflowing to trigger policy execution
        self.means = np.ones(n) * 1.0
        self.std = np.ones(n) * 0.1

@pytest.fixture
def mock_engine_data():
    n_bins = 5
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    # Create fake new_data (DataFrame)
    new_data = pd.DataFrame({
        "#bin": range(n_bins + 1),
        "Stock": [0.0] * (n_bins + 1),
        "Accum_Rate": [0.0] * (n_bins + 1)
    })

    # Create fake coords
    coords = pd.DataFrame({
        "Lat": [0.0] * (n_bins + 1),
        "Lng": [0.0] * (n_bins + 1)
    })

    return {
        "bins": MockBins(n_bins),
        "distance_matrix": dist_matrix,
        "waste_type": "plastic",
        "area": "test_area",
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
        "distancesC": np.zeros((n_bins+1, n_bins+1)),
        "n_vehicles": 2,
        "model_env": None,
        "run_tsp": False,
        "new_data": new_data,
        "coords": coords,
        "graph_size": n_bins
    }

def test_bcp_engine_override(mock_engine_data):
    # Patch where it is used: policy_bcp module
    with patch.object(policy_bcp_module, "run_bcp") as mock_run:
        mock_run.return_value = ([[1]], 10.0)

        policy = BCPPolicy()

        # Call with engine override
        policy.execute(policy="policy_bcp", engine="gurobi", **mock_engine_data)

        assert mock_run.called
        args, kwargs = mock_run.call_args
        config_arg = args[5]
        assert config_arg["bcp_engine"] == "gurobi"

def test_vrpp_engine_override(mock_engine_data):
    # Retrieve the module from sys.modules to avoid package shadowing confusion
    import sys
    # Ensure it's loaded
    import logic.src.policies.policy_vrpp
    policy_vrpp_module = sys.modules["logic.src.policies.policy_vrpp"]

    with patch.object(policy_vrpp_module, "run_vrpp_optimizer") as mock_opt, \
         patch.object(policy_vrpp_module, "load_area_and_waste_type_params") as mock_load:

        # Mock loader returns Q, R, B, C, V (floats)
        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)

        # It needs to return routes, profit, cost
        mock_opt.return_value = ([1, 0], 10.0, 5.0)

        policy = VRPPPolicy()

        # Default call: "gurobi_vrpp_1.0"
        policy.execute(policy="gurobi_vrpp_1.0", **mock_engine_data)

        assert mock_opt.called
        args, kwargs = mock_opt.call_args
        # kwargs["optimizer"] check
        assert kwargs["optimizer"] == "gurobi"

        # Override call: engine="hexaly"
        policy.execute(policy="gurobi_vrpp_1.0", engine="hexaly", **mock_engine_data)

        args, kwargs = mock_opt.call_args
        assert kwargs["optimizer"] == "hexaly"

def test_hgs_engine_override(mock_engine_data):
    # Patch usage in policy_hgs
    with patch.object(policy_hgs_module, "run_hgs") as mock_run:
        mock_run.return_value = ([[1]], 100.0, 50.0)

        policy = HGSPolicy()

        # Test Default
        policy.execute(policy="policy_hgs_1.0", **mock_engine_data)
        assert mock_run.called
        args, kwargs = mock_run.call_args
        # run_hgs(..., values, ...) -> values is 6th arg (index 5)
        config_arg = args[5]
        assert "engine" not in config_arg or config_arg["engine"] is None

        # Test Override
        policy.execute(policy="policy_hgs_1.0", engine="pyvrp", **mock_engine_data)
        args, kwargs = mock_run.call_args
        config_arg = args[5]
        assert config_arg["engine"] == "pyvrp"

def test_alns_engine_override(mock_engine_data):
    # Patch usage in policy_alns
    with patch.object(policy_alns_module, "run_alns") as mock_run:
        # Return properly nested routes list
        mock_run.return_value = ([[1]], 10.0, 5.0)

        policy = ALNSPolicy()

        # Test Default
        policy.execute(policy="policy_alns_1.0", **mock_engine_data)
        assert mock_run.called
        args, kwargs = mock_run.call_args
        # run_alns(dist_matrix, demands, capacity, R, C, values)
        # indexes: 0, 1, 2, 3, 4, 5
        values_arg = args[5]
        assert "engine" not in values_arg or values_arg.get("engine") is None

        # Test Override
        policy.execute(policy="policy_alns_1.0", engine="ortools", **mock_engine_data)
        args, kwargs = mock_run.call_args
        values_arg = args[5]
        assert values_arg["engine"] == "ortools"

def test_sans_execution(mock_engine_data):
    # Patch policy_lookahead_sans in its new home
    with patch.object(policy_sans_module, "policy_lookahead_sans") as mock_sans, \
         patch.object(policy_sans_module, "load_area_and_waste_type_params") as mock_load:

        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)
        mock_sans.return_value = ([[1, 0]], 10.0, 5.0)

        policy = SANSPolicy()
        policy.execute(policy="policy_sans_1.0", **mock_engine_data)

        assert mock_sans.called

def test_lac_execution(mock_engine_data):
    # Patch find_solutions in LAC module
    with patch.object(policy_lac_module, "find_solutions") as mock_find, \
         patch.object(policy_lac_module, "load_area_and_waste_type_params") as mock_load:

        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)
        mock_find.return_value = ([[1, 0]], 10.0, 5.0)

        policy = LACPolicy()
        policy.execute(policy="policy_lac_a_1.0", **mock_engine_data)

        assert mock_find.called
