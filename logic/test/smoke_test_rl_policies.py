import numpy as np

def test_rl_alns_solver():
    print("\n=== Testing RLALNSSolver ===")
    from logic.src.policies.rl_alns.solver import RLALNSSolver
    from logic.src.policies.rl_alns.params import RLALNSParams

    # Mock Data
    dist_matrix = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]])
    wastes = {1: 10, 2: 20}
    capacity = 50.0
    R = 1.0
    C = 1.0

    # Test several algorithms
    algos = ["ucb1", "q_learning", "sarsa"]

    for algo in algos:
        print(f"Testing with algorithm: {algo}")
        params = RLALNSParams(
            rl_algorithm=algo,
            max_iterations=5,
            n_operators=6,
            start_temp=100.0,
            cooling_rate=0.9
        )

        solver = RLALNSSolver(dist_matrix, wastes, capacity, R, C, params, seed=42)
        routes, profit, cost = solver.solve()

        print(f"  Result: Routes={routes}, Profit={profit:.2f}, Cost={cost:.2f}")
        assert profit is not None
        assert isinstance(routes, list)

def test_ahvpl_rl_solver():
    print("\n=== Testing AHVPLRLSolver ===")
    from logic.src.policies.rl_ahvpl.solver import RLAHVPLSolver as AHVPLRLSolver
    from logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params import RLAHVPLParams
    from logic.src.policies.ant_colony_optimization.k_sparse_aco.params import ACOParams

    # Mock Data
    dist_matrix = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]])
    wastes = {1: 10, 2: 20}
    capacity = 50.0
    R = 1.0
    C = 1.0

    aco_params = ACOParams(max_iterations=5, n_ants=2)
    rl_params = RLAHVPLParams(
        bandit_algorithm="linucb",
        qlearning_epsilon=0.1
    )

    solver = AHVPLRLSolver(dist_matrix, wastes, capacity, R, C, aco_params, rl_params, seed=42)
    routes, profit, cost = solver.solve()

    print(f"  Result: Routes={routes}, Profit={profit:.2f}, Cost={cost:.2f}")
    assert profit is not None
    assert isinstance(routes, list)

if __name__ == "__main__":
    try:
        test_rl_alns_solver()
        test_ahvpl_rl_solver()
        print("\nAll RL smoke tests passed!")
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
