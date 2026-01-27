
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from logic.src.policies.adapters import PolicyRegistry
from logic.src.policies.policy_tsp import TSPPolicy
from logic.src.policies.policy_cvrp import CVRPPolicy

class MockBins:
    def __init__(self, n=5):
        self.n = n
        self.c = np.ones(n) * 10.0 # 10.0 fill
        self.collectlevl = 90.0

@pytest.fixture
def mock_params():
    n_bins = 5
    # (N+1)x(N+1) matrix for 5 bins + depot
    dist_matrix = np.ones((n_bins + 1, n_bins + 1), dtype=np.int32)
    distancesC = dist_matrix

    return {
        "bins": MockBins(n_bins),
        "distancesC": distancesC,
        "waste_type": "plastic",
        "area": "test_area",
        "n_vehicles": 2,
        "coords": MagicMock()
    }

def test_tsp_policy(mock_params):
    with patch("logic.src.policies.policy_tsp.load_area_and_waste_type_params") as mock_load, \
         patch("logic.src.policies.policy_tsp.find_route") as mock_find, \
         patch("logic.src.policies.policy_tsp.get_multi_tour") as mock_get_multi:

        # Mock load: cap, rev, dens, exp, vol
        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)

        # Mock find_route (TSP)
        # Returns simple tour [0, 1, 2, 3, 4, 5, 0]
        mock_find.return_value = [0, 1, 2, 3, 4, 5, 0]

        # Mock get_multi_tour (Split logic for 1 vehicle)
        # Assume it just returns the tour if valid
        mock_get_multi.return_value = [0, 1, 2, 3, 4, 5, 0]

        policy = PolicyRegistry.get("tsp")()
        assert isinstance(policy, TSPPolicy)

        tour, cost, extra = policy.execute(policy="tsp", **mock_params)

        assert tour == [0, 1, 2, 3, 4, 5, 0]
        # Valid cost (mocked distances are 1.0 everywhere except diagonal)
        # 6 steps. 1.0 * 6 = 6.0
        assert cost == 6.0

        # Verify find_route called with all bins
        args, _ = mock_find.call_args
        assert args[1] == [1, 2, 3, 4, 5]

def test_cvrp_policy(mock_params):
    with patch("logic.src.policies.policy_cvrp.load_area_and_waste_type_params") as mock_load, \
         patch("logic.src.policies.policy_cvrp.find_routes") as mock_find:

        mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1.0)

        # Mock find_routes (CVRP)
        mock_find.return_value = [0, 1, 2, 0, 3, 4, 5, 0] # 2 routes

        policy = PolicyRegistry.get("cvrp")()
        assert isinstance(policy, CVRPPolicy)

        tour, cost, extra = policy.execute(policy="cvrp", **mock_params)

        assert tour == [0, 1, 2, 0, 3, 4, 5, 0]
        # 7 edges. 1.0 each (except 0-0 dep-dep? no dep-dep in this list)
        # 0->1, 1->2, 2->0, 0->3, 3->4, 4->5, 5->0
        assert cost == 7.0

        # Verify n_vehicles passed
        args, _ = mock_find.call_args
        assert args[4] == 2 # n_vehicles
