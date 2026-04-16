"""
Unit tests for the Scenario-Tree Extensive Form (ST-EF) policy.
"""

import numpy as np
import pytest
from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form import ScenarioTreeExtensiveFormPolicy
from logic.src.policies.route_construction.exact_and_decomposition_solvers.scenario_tree_extensive_form.st_ef_engine import GUROBI_AVAILABLE


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available for ST-EF solver")
def test_st_ef_basic():
    """Test ST-EF policy on a tiny instance."""
    n_nodes = 3
    # 0 -> Depot, 1/2 -> Customers
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    # Initial wastes at the nodes (Day 0)
    wastes = {1: 50.0, 2: 50.0}
    capacity = 100.0

    # ST-EF Configuration
    config = {
        "num_days": 2,
        "num_realizations": 2,
        "mean_increment": 0.3,
        "time_limit": 30.0,
        "mip_gap": 0.01,
        "waste_weight": 1.0,
        "cost_weight": 0.1,
        "overflow_penalty": 10.0,
        "use_mtz": True,
        "seed": 42
    }

    policy = ScenarioTreeExtensiveFormPolicy(config=config)

    # Simulation-like call
    # The policy expects these keys in kwargs
    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "mandatory": [],
        "R": 1.0,
        "C": 0.1,
        "number_vehicles": 1,
        "coords": np.array([[0,0], [1,1], [2,2]]),
        "day": 1
    }

    tour, cost, extra = policy.execute(**kwargs)

    # Basic validity checks
    assert isinstance(tour, list)
    assert len(tour) >= 2
    assert tour[0] == 0
    assert tour[-1] == 0
    assert cost >= 0
    assert "expected_value" in extra

    print(f"\nExtracted Tour (Day 1): {tour}")
    print(f"Total Cost: {cost:.2f}")
    print(f"Expected Value: {extra['expected_value']:.2f}")

@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available for ST-EF solver")
def test_st_ef_no_waste():
    """Test ST-EF policy when there's nothing to collect."""
    n_nodes = 3
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.5],
        [1.0, 1.5, 0.0]
    ])

    wastes = {1: 0.0, 2: 0.0}
    capacity = 100.0

    config = {
        "num_days": 2,
        "num_realizations": 2,
        "mean_increment": 0.0, # NO GROWTH
        "time_limit": 10.0,
        "use_mtz": True,
        "seed": 42
    }

    policy = ScenarioTreeExtensiveFormPolicy(config=config)

    kwargs = {
        "distance_matrix": dist_matrix,
        "wastes": wastes,
        "capacity": capacity,
        "day": 1
    }

    tour, cost, extra = policy.execute(**kwargs)

    # Should probably stay at depot since cost > gain (gain is 0)
    assert tour == [0, 0]
    assert cost == 0.0
