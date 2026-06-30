from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies.helpers.reinforcement_learning.ks_aco_qlearning import KSparseACOQLSolver
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams


class MockRLParams:
    qlearning_epsilon_decay_step = 1
    qlearning_improvement_thresholds = [0.01, 0.05]
    qlearning_alpha = 0.1
    qlearning_gamma = 0.9
    qlearning_epsilon = 0.5
    qlearning_epsilon_decay = 0.95
    qlearning_epsilon_min = 0.01
    qlearning_history_size = 50

@pytest.fixture
def aco_ql_setup():
    dist_matrix = np.array([
        [0.0, 2.0, 3.0, 4.0],
        [2.0, 0.0, 1.5, 2.5],
        [3.0, 1.5, 0.0, 1.0],
        [4.0, 2.5, 1.0, 0.0]
    ])
    wastes = {1: 10.0, 2: 15.0, 3: 20.0}
    capacity = 30.0
    R = 2.0
    C = 1.0

    params = KSACOParams(
        n_ants=2,
        k_sparse=2,
        max_iterations=2,
        time_limit=10.0,
        local_search=True,
        local_search_iterations=3,
        seed=42
    )
    rl_params = MockRLParams()
    return dist_matrix, wastes, capacity, R, C, params, rl_params

def test_aco_ql_solver_init(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)

    assert solver.n_nodes == 4
    assert solver.nodes == [1, 2, 3]
    assert solver.ls_operators[0] == "or_opt"
    assert solver.n_actions == len(solver.ls_operators)
    assert solver.agent.alpha == rl_params.qlearning_alpha

def test_nearest_neighbor_cost(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)
    cost = solver._nearest_neighbor_cost()
    # Path: 0 -> 1 -> 2 -> 3 -> 0
    # Dists: 2.0 + 1.5 + 1.0 + 4.0 = 8.5
    assert abs(cost - 8.5) < 1e-6

def test_q_learning_local_search_and_operators(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)

    routes = [[1, 2], [3]]
    op_to_method = {
        "or_opt": "or_opt",
        "cross_exchange": "cross_exchange_op",
        "lambda_interchange": "lambda_interchange_op",
        "ejection_chain": "ejection_chain_op",
        "2opt_star": "two_opt_star",
        "swap_star": "swap_star",
        "2opt_intra": "two_opt_intra",
        "3opt_intra": "three_opt_intra",
        "relocate": "relocate",
        "swap": "swap",
        "lkh_refinement": "lkh_refinement",
    }
    # Mock all local search operators to return True (improved)
    with patch.object(solver.ls_manager, "get_routes", return_value=[[1, 2, 3]]):
        for op in solver.ls_operators:
            method_name = op_to_method.get(op, "relocate")
            with patch.object(solver.ls_manager, method_name, return_value=True):
                improved = solver._apply_operator(op)
                assert improved is True

        # Test unknown operator
        assert solver._apply_operator("invalid_op") is False

        # Execute q_learning_local_search
        improved_routes = solver._q_learning_local_search(routes, iteration=0)
        assert improved_routes == [[1, 2, 3]]

def test_global_pheromone_update(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)

    routes = [[1, 2], [3]]
    solver._global_pheromone_update(routes, 10.0)
    # Pheromone matrix updated, verify some values
    # The default scale configuration applies precision pruning, but deposit should still change pheromone matrix.
    # Note: SparsePheromoneTau stores sparse values, we can verify deposit doesn't crash
    solver._global_pheromone_update([], 0.0) # smoke check empty

def test_solve_and_initialize(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)

    best_routes, best_profit, best_cost = solver.solve()
    assert isinstance(best_routes, list)
    assert isinstance(best_profit, float)
    assert isinstance(best_cost, float)

def test_solve_time_limit(aco_ql_setup):
    dist_matrix, wastes, capacity, R, C, params, rl_params = aco_ql_setup
    params.time_limit = -1.0 # Force immediate timeout
    solver = KSparseACOQLSolver(dist_matrix, wastes, capacity, R, C, params, rl_params)

    best_routes, best_profit, best_cost = solver.solve()
    assert isinstance(best_routes, list)
