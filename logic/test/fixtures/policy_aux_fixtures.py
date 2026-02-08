"""
Fixtures for policy auxiliary function tests (Look-ahead, HGS, ALNS).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def policies_routes_setup():
    """Standard routes setup for move/swap tests."""
    r1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    r2 = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0]
    return [r1[:], r2[:]]  # Return copies


@pytest.fixture
def policies_vpp_data():
    """Dataframe for VRPP/Solution tests."""
    data = pd.DataFrame(
        {
            "#bin": [0, 1, 2, 3],  # 0 is depot
            "Stock": [0, 50, 50, 50],
            "Accum_Rate": [0, 10, 10, 10],
            "Lng": [10.0, 10.05, 10.15, 10.25],
            # Zones: [10.0, 10.083), [10.083, 10.166), [10.166, 10.25]
        }
    )
    return data


@pytest.fixture
def policies_bins_coords():
    """Coords dataframe."""
    return pd.DataFrame({"Lng": [10.0, 10.05, 10.15, 10.25]})


@pytest.fixture
def policies_dist_matrix():
    """Simple distance matrix."""
    dist_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            dist_matrix[i, j] = abs(i - j)
    return dist_matrix


@pytest.fixture
def policies_solution_values():
    """Values dict for find_solutions."""
    return {"vehicle_capacity": 100, "E": 1.0, "B": 1.0, "perc_bins_can_overflow": 0.0}


class MockLS:
    """Mock LocalSearch object for operator testing."""
    def __init__(self, routes, dist_matrix, waste, capacity):
        self.routes = routes
        self.d = dist_matrix
        self.waste = waste
        self.Q = capacity
        self.C = 1.0 # Minimize distance
        # Precompute initial loads
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]
        # Mocking update_map to avoid real logic dependency during fixture setup if not needed
        # But if tests depend on it... tests in test_hgs_operators use a Mock for it.
        # We can set it to a MagicMock if we import it, or just a dummy function.
        # tests used: self._update_map = MagicMock()
        from unittest.mock import MagicMock
        self._update_map = MagicMock()

    def _calc_load_fresh(self, r):
        return sum(self.waste.get(x, 0) for x in r)

    def _get_load_cached(self, ri):
        return self.route_loads[ri]


@pytest.fixture
def mock_ls():
    """
    Creates a basic mock LS state.
    Nodes: 0 (Depot), 1, 2, 3, 4
    Routes:
      0: [1, 2]
      1: [3, 4]
    """
    routes = [[1, 2], [3, 4]]
    # 5 nodes: 0..4
    # Simple distance: 1.0 between adjacent nodes in numeric order, 10 otherwise
    d = np.full((5, 5), 10.0)
    np.fill_diagonal(d, 0.0)

    # Path 0->1->2->0: 1->2 is close
    d[0, 1] = d[1, 0] = 5.0
    d[1, 2] = d[2, 1] = 1.0
    d[2, 0] = d[0, 2] = 5.0

    # Path 0->3->4->0
    d[0, 3] = d[3, 0] = 5.0
    d[3, 4] = d[4, 3] = 1.0
    d[4, 0] = d[0, 4] = 5.0

    # Neighbors for cross moves
    # 2 is close to 3?
    d[2, 3] = d[3, 2] = 1.0

    waste = {1: 10, 2: 10, 3: 10, 4: 10}
    capacity = 100.0

    return MockLS(routes, d, waste, capacity)
