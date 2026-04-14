import pytest
import numpy as np
from logic.src.policies.logic_based_benders_decomposition.policy_lbbd import LBBDPolicy
from logic.src.policies.logic_based_benders_decomposition.lbbd_engine import LBBDEngine

def test_lbbd_basic():
    # Toy problem: 3 nodes (0=depot, 1, 2)
    # Distances: (0,1)=1, (0,2)=1, (1,2)=1.5
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    # Initial wastes
    wastes = {1: 0.1, 2: 0.1}

    config = {
        "num_days": 2,
        "max_iterations": 5,
        "seed": 42,
        "waste_weight": 10.0,
        "cost_weight": 1.0,
        "overflow_penalty": 100.0,
        "mean_increment": 0.5
    }

    policy = LBBDPolicy(config=config)

    # Mocking kwargs for execute
    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 100.0, # Large capacity
        "day": 1
    }

    route, cost, info = policy.execute(**kwargs)

    assert len(route) >= 2
    assert route[0] == 0
    assert route[-1] == 0
    assert "expected_value" in info
    assert "stats" in info
    assert info["stats"]["iterations"] > 0

def test_lbbd_infeasibility_nogood():
    # Make nodes very far or capacity very small to trigger nogood cuts
    dist_matrix = np.array([
        [0.0, 10.0, 10.0],
        [10.0, 0.0, 1.5],
        [10.0, 1.5, 0.0]
    ])

    # Bins are full
    wastes = {1: 0.9, 2: 0.9}

    # Capacity is small - can only fit one bin
    # But LBBD Master might try to fit both if it doesn't know the exact routing cost
    # Actually, Master Problem capacity is sum(q) <= 1.0
    # In LBBD, the Subproblem checks traveling distance/time limits.
    # Let's say we have a time limit L in Subproblem.

    config = {
        "num_days": 1,
        "max_iterations": 10,
        "cost_weight": 1.0,
        "subproblem_timeout": 5.0
    }

    policy = LBBDPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": 1.0,
        "day": 1
    }

    route, cost, info = policy.execute(**kwargs)
    assert route[0] == 0
    # It should still find a route, just maybe only visiting one or zero if profit < cost
