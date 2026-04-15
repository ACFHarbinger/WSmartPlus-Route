import numpy as np


def test_rl_alns_solver():
    print("\n=== Testing RLALNSSolver ===")
    from logic.src.policies.reinforcement_learning_adaptive_large_neighborhood_search.params import RLALNSParams
    from logic.src.policies.reinforcement_learning_adaptive_large_neighborhood_search.solver import RLALNSSolver

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
        from logic.src.configs.policies.helpers import RLConfig, BanditConfig
        rl_config = RLConfig(
            agent_type="bandit",
            bandit=BanditConfig(algorithm=algo)
        )
        params = RLALNSParams(
            rl_config=rl_config,
            max_iterations=5,
            start_temp=100.0,
            cooling_rate=0.9,
            seed=42
        )

        solver = RLALNSSolver(dist_matrix, wastes, capacity, R, C, params)
        routes, profit, cost = solver.solve()

        print(f"  Result: Routes={routes}, Profit={profit:.2f}, Cost={cost:.2f}")
        assert profit is not None
        assert isinstance(routes, list)

def test_ahvpl_rl_solver():
    print("\n=== Testing AHVPLRLSolver ===")
    from logic.src.policies.ant_colony_optimization_k_sparse.params import KSACOParams
    from logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params import (
        RLAHVPLParams,
    )
    from logic.src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl import RLAHVPLSolver as AHVPLRLSolver

    # Mock Data
    dist_matrix = np.array([[0, 10, 20], [10, 0, 15], [20, 15, 0]])
    wastes = {1: 10, 2: 20}
    capacity = 50.0
    R = 1.0
    C = 1.0

    aco_params = KSACOParams(max_iterations=5, n_ants=2)
    from logic.src.configs.policies.helpers import RLConfig, BanditConfig, LinUCBConfig
    rl_config = RLConfig(
        agent_type="bandit",
        bandit=BanditConfig(algorithm="linucb"),
        contextual=LinUCBConfig(alpha=0.1)
    )
    rl_params = RLAHVPLParams(
        rl_config=rl_config,
        aco_params=aco_params
    )

    solver = AHVPLRLSolver(dist_matrix, wastes, capacity, R, C, rl_params)
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
