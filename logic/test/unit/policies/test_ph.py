"""
Unit tests for the Progressive Hedging (PH) policy.
"""

import numpy as np
import pytest
from logic.src.configs.policies import PHConfig
from logic.src.policies.route_construction.exact_and_decomposition_solvers.progressive_hedging import ProgressiveHedgingPolicy
from logic.src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc import GUROBI_AVAILABLE


@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available for subproblems")
def test_ph_convergence_small():
    """Test Progressive Hedging on a small stochastic instance."""
    n_nodes = 5
    # Simple star distance matrix
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes):
        dist_matrix[0, i] = 1.0  # Depot to node i
        dist_matrix[i, 0] = 1.0
        for j in range(1, n_nodes):
            if i != j:
                dist_matrix[i, j] = 1.5

    # Define 3 scenarios with different node wastes
    # Scenario 1: Nodes 1 and 2 are full
    # Scenario 2: Nodes 3 and 4 are full
    # Scenario 3: All nodes have medium fill
    scenarios = [
        {1: 50.0, 2: 50.0, 3: 0.0, 4: 0.0},
        {1: 0.0, 2: 0.0, 3: 50.0, 4: 50.0},
        {1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0},
    ]

    capacity = 100.0  # Can visit 2 nodes fully or 3 nodes partially
    revenue = 2.0
    cost_unit = 1.0

    # Progressive Hedging Configuration
    values = {
        "rho": 1.0,
        "max_iterations": 20,
        "convergence_tol": 0.05,
        "sub_solver": "bc",
        "verbose": True,
        "seed": 42
    }
    config = PHConfig(
        rho=values["rho"],
        max_iterations=values["max_iterations"],
        convergence_tol=values["convergence_tol"],
        sub_solver=values["sub_solver"],
        verbose=values["verbose"],
        seed=values["seed"]
    )

    policy = ProgressiveHedgingPolicy(config=config)

    # We use _run_solver directly to pass scenarios
    routes, expected_profit, total_dist = policy._run_solver(
        sub_dist_matrix=dist_matrix,
        sub_wastes=scenarios[0], # Base wastes, but scenarios will override
        capacity=capacity,
        revenue=revenue,
        cost_unit=cost_unit,
        values=values,
        scenarios=scenarios,
        mandatory_nodes=[]
    )

    # Basic validity checks
    assert len(routes) > 0
    assert expected_profit > 0
    assert total_dist > 0

    # Check that the routes contain customers only (factory expectation)
    # The depot (0) is added by the adapter's _map_tour_to_global method.
    for route in routes:
        for node in route:
            assert node != 0
            assert node < n_nodes

    print(f"\nFinal Expected Profit: {expected_profit:.2f}")
    print(f"Final Routes: {routes}")
