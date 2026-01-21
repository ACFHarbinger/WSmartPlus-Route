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
