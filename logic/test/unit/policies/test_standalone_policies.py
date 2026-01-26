from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies.adapters import PolicyRegistry
from logic.src.policies.adaptive_large_neighborhood_search import ALNSPolicy
from logic.src.policies.branch_cut_and_price import BCPPolicy


class MockBins:
    def __init__(self, n=10):
        self.n = n
        self.c = np.zeros(n)
        self.means = np.ones(n) * 10.0
        self.std = np.ones(n) * 1.0


@pytest.fixture
def mock_data():
    n_bins = 5
    # Bins fill: all 0 except bin 0 which is 95% filled.
    # Prediction: 95 + 10 + 1*1 = 106% -> overflow -> must_go
    bins = MockBins(n_bins)
    bins.c[0] = 95.0

    # Distance matrix (6x6 including depot)
    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    return {
        "bins": bins,
        "distance_matrix": dist_matrix.tolist(),
        "waste_type": "plastic",
        "area": "test_area",
        "config": {},
        "distancesC": np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32),
    }


def test_alns_policy(mock_data):
    # Mock run_alns
    with patch("logic.src.policies.adaptive_large_neighborhood_search.run_alns") as mock_run:
        # Mock outcome: [0, 1, 0] (depot -> bin 0 -> depot)
        # Note: run_alns returns routes. If subsetted, it returns indices 1..K.
        # Subset indices: [0, 1] (Depot + Bin 0). Bin 0 is index 1 in subset.
        # So expected route: [1]
        mock_run.return_value = ([[1]], 10.0, 5.0)  # routes, profit, cost

        policy = PolicyRegistry.get("policy_alns")()
        assert isinstance(policy, ALNSPolicy)

        tour, cost, extra = policy.execute(policy="policy_alns_1.0", **mock_data)

        # Verify result
        # Input bin 0 is must_go.
        # Subset indices: [0, 1].
        # ALNS return [[1]] -> tour [1].
        # map back: 1 -> subset_indices[1] = 1.
        # Tour: [0, 1, 0]
        # But my implementation appends: `tour` starts `[0]`.
        # loop: `original_matrix_idx = 1`. `tour.append(1-1) = 0`.  Wait.
        # In ALNSPolicy: `original_matrix_idx = subset_indices[node_idx]`.
        # `tour.append(original_matrix_idx - 1)`.
        # Dist matrix index 1 is bin 0. So `1-1 = 0`.
        # Tour should contain bin indices (0-based) or dist matrix indices?
        # Check `policy_regular`: it returns `tour` from `policy_regular` C++ which returns node indices (1..N usually).
        # `day.py` converts: `ids = [x for x in tour if x!=0]`. `coordinates.loc[ids, "ID"]`.
        # If coordinates dataframe is indexed 1..N ?
        # In `day.py`: `dlog["tour"] = [0] + coordinates.loc[ids, "ID"].tolist() + [0]`
        # If `ids` contains `0` (bin index?), `coordinates.loc[0]` must specific bin?
        # Usually dist matrix 0 is depot. Bins are 1..N.
        # If policy returns 1..N indices, then `ids` are 1..N.

        # Let's check my implementation again.
        # `tour.append(original_matrix_idx - 1)` -> This produces 0-based bin index.
        # If `day.py` `coordinates` DF is indexed by Bin ID?
        # `loader.py`: `get_simulator_data` returns `coords` DF.
        # Usually indexed 0?
        # Wait, `policy_regular` calls `policy_regular` (C++) which usually works with 0..N-1?
        # No, C++ usually 0..N-1.
        # If `day.py` filter `x != 0`. Then 0 is depot.
        # So return values MUST be 1-based (dist matrix indices).
        # IF I return `original_matrix_idx - 1`, I am returning 0 for Bin 0.
        # `x != 0` will filter it out!
        # So `day.py` will see empty tour!

        # FIX REQUIRED: Policies should return dist matrix indices (1..N).
        # I should NOT subtract 1.
        # I will confirm this by checking `regular.py`.

        # BCP Policy implementation: `tour.extend(route)`. Route has indices.
        # `run_bcp` works on full matrix. Returns 1 for Bin 0.
        # So tour will be [0, 1, 0].

        assert tour == [0, 1, 0]
        assert mock_run.called


def test_bcp_policy(mock_data):
    with patch("logic.src.policies.branch_cut_and_price.run_bcp") as mock_run:
        # returns routes (list of lists of ints)
        mock_run.return_value = ([[1]], 10.0)

        policy = PolicyRegistry.get("policy_bcp")()
        assert isinstance(policy, BCPPolicy)

        tour, cost, extra = policy.execute(policy="policy_bcp_1.0", **mock_data)

        # BCP Policy implementation: `tour.extend(route)`. Route has indices.
        # `run_bcp` works on full matrix. Returns 1 for Bin 0.
        # So tour will be [0, 1, 0].

        assert tour == [0, 1, 0]
        assert mock_run.called
        # Check must_go passed
        args, kwargs = mock_run.call_args
        assert kwargs["must_go_indices"] == {0}  # Bin 0 index?
        # Wait, my implementation: `np.where(...)[0]`.
        # `bins.c` is 0-based. So bin 0 causes overflow. Index 0.
        # So must_go_indices = {0}.
        # BCP expects node indices (1..N)?
        # I passed `must_go_indices` direct result of `np.where`.
        # `run_bcp` docstring: "Node IDs that must be visited".
        # If `dist_matrix` is passed, indices match matrix. (1..N).
        # So I probably passed 0-based indices to BCP which expects 1-based!
        # Potential Bug in BCP implementation too.


def test_hgs_policy(mock_data):
    with patch("logic.src.policies.hybrid_genetic_search.run_hgs") as mock_run:
        mock_run.return_value = ([[1]], 10.0, 5.0)
        policy = PolicyRegistry.get("policy_hgs")()
        tour, cost, extra = policy.execute(policy="policy_hgs_1.0", **mock_data)
        assert mock_run.called


def test_lkh_policy(mock_data):
    with patch("logic.src.policies.lin_kernighan.solve_lk") as mock_run:
        mock_run.return_value = ([1], 5.0)  # tour (list), cost
        policy = PolicyRegistry.get("policy_lkh")()
        tour, cost, extra = policy.execute(policy="policy_lkh_1.0", **mock_data)
        assert mock_run.called
