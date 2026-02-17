import pytest
import numpy as np
from logic.src.policies.hybrid_volleyball_premier_league.hvpl import HVPLSolver
from logic.src.policies.hybrid_volleyball_premier_league.params import HVPLParams

def test_hvpl_solver():
    # Setup simple problem: 3 bins + depot
    dist_matrix = np.array([
        [0.0, 10.0, 20.0, 10.0],
        [10.0, 0.0, 10.0, 20.0],
        [20.0, 10.0, 0.0, 10.0],
        [10.0, 20.0, 10.0, 0.0]
    ])
    demands = {1: 10.0, 2: 10.0, 3: 10.0}
    capacity = 50.0
    R = 10.0
    C = 1.0

    # Use very small params for fast test
    params = HVPLParams(
        n_teams=3,
        max_iterations=2,
        sub_rate=0.3,
        time_limit=5.0
    )
    # Adjust nested params for speed
    params.alns_params.max_iterations = 5
    params.aco_params.n_ants = 2

    solver = HVPLSolver(dist_matrix, demands, capacity, R, C, params)

    routes, profit, cost = solver.solve()

    # Basic sanity checks
    assert isinstance(routes, list)
    assert len(routes) > 0
    assert isinstance(profit, float)
    assert isinstance(cost, float)

    # Check that all nodes were visited (since it's a small problem)
    visited = set()
    for route in routes:
        for node in route:
            visited.add(node)
    assert visited == {1, 2, 3}

if __name__ == "__main__":
    test_hvpl_solver()
