import os
import sys

import numpy as np


def test_alns_sarsa_operator_integration():
    """Verify that all unstringing operators are correctly registered as destroy operators."""
    # Add project root to sys.path if not there
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from logic.src.policies.adaptive_large_neighborhood_search.params import ALNSParams
    from logic.src.policies.helpers.reinforcement_learning.alns_sarsa import ALNSSARSASolver
    from logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params import (
        RLAHVPLParams,
    )

    # Mock Data
    dist_matrix = np.zeros((10, 10))
    wastes = {i: 1.0 for i in range(1, 10)}
    capacity = 10.0
    R, C = 10.0, 1.0

    alns_params = ALNSParams(
        max_iterations=1,
        start_temp=100.0,
        cooling_rate=0.95,
        min_removal=1,
        max_removal_pct=0.3,
        time_limit=1.0,
    )

    rl_params = RLAHVPLParams()

    # We might need to mock calculate_cost if it fails due to missing logic
    # But for initialization we just need the solver instance
    solver = ALNSSARSASolver(
        dist_matrix, wastes, capacity, R, C, alns_params, rl_params
    )

    expected_destroy_names = [
        "Random", "Worst", "Cluster", "Shaw", "String",
        "Unstring-I", "Unstring-II", "Unstring-III", "Unstring-IV"
    ]

    print(f"Destroy operators found: {solver.destroy_names}")

    assert len(solver.destroy_ops) == 9
    assert len(solver.destroy_names) == 9
    assert solver.destroy_names == expected_destroy_names

    # Check that they are callable
    for op in solver.destroy_ops:
        assert callable(op)

if __name__ == "__main__":
    try:
        test_alns_sarsa_operator_integration()
        print("Verification smoke test passed!")
    except Exception as e:
        print(f"Verification smoke test failed: {e}")
        exit(1)
