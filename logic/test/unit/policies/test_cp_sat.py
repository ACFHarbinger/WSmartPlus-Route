import pytest
import numpy as np
from logic.src.policies.cp_sat.policy_cp_sat import CPSATPolicy

def test_cp_sat_basic():
    # Toy problem: 3 nodes (0=depot, 1, 2)
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    wastes = {1: 0.1, 2: 0.1}

    config = {
        "num_days": 2,
        "time_limit": 30.0,
        "seed": 42,
        "waste_weight": 10.0,
        "cost_weight": 1.0,
        "overflow_penalty": 100.0,
        "mean_increment": 0.5
    }

    policy = CPSATPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 100.0,
        "day": 1
    }

    route, cost, info = policy.execute(**kwargs)

    assert len(route) >= 2
    assert route[0] == 0
    assert route[-1] == 0
    assert "expected_value" in info
    assert "stats" in info
    assert info["stats"]["status"] in ("OPTIMAL", "FEASIBLE")

def test_cp_sat_multi_day():
    # Verify that it plans for future days
    dist_matrix = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ])

    # Bin 1 is almost full
    wastes = {1: 0.9}

    config = {
        "num_days": 1,
        "cost_weight": 1.0,
        "waste_weight": 10.0
    }

    policy = CPSATPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 1.0,
        "day": 1
    }

    route, cost, info = policy.execute(**kwargs)
    # Should visit bin 1
    assert 1 in route
