import pytest
import numpy as np
from logic.src.policies.harmony_search.solver import HSSolver
from logic.src.policies.harmony_search.params import HSParams
from logic.src.policies.sine_cosine_algorithm.solver import SCASolver
from logic.src.policies.sine_cosine_algorithm.params import SCAParams
from logic.src.policies.guided_indicators_hyper_heuristic.gihh import GIHHSolver
from logic.src.policies.guided_indicators_hyper_heuristic.params import GIHHParams

@pytest.fixture
def tiny_vrpp():
    # 3 nodes + depot
    dist_matrix = np.array([
        [0, 10, 10, 10],
        [10, 0, 5, 15],
        [10, 5, 0, 15],
        [10, 15, 15, 0]
    ])
    wastes = {1: 10.0, 2: 10.0, 3: 10.0}
    capacity = 15.0
    R, C = 1.0, 0.1
    return dist_matrix, wastes, capacity, R, C

def test_hs_refinement(tiny_vrpp):
    dist, wastes, cap, R, C = tiny_vrpp
    params = HSParams(max_iterations=10, hm_size=5, BW=0.1)
    solver = HSSolver(dist, wastes, cap, R, C, params, seed=42)
    routes, profit, cost = solver.solve()
    assert len(routes) > 0
    assert profit > 0

def test_sca_refinement(tiny_vrpp):
    dist, wastes, cap, R, C = tiny_vrpp
    params = SCAParams(max_iterations=10, pop_size=5)
    solver = SCASolver(dist, wastes, cap, R, C, params, seed=42)
    routes, profit, cost = solver.solve()
    assert len(routes) > 0
    assert profit > 0

def test_gihh_refinement(tiny_vrpp):
    dist, wastes, cap, R, C = tiny_vrpp
    params = GIHHParams(max_iterations=10, restarts=1)
    solver = GIHHSolver(dist, wastes, cap, R, C, params, seed=42)
    routes, profit, cost = solver.solve()
    assert len(routes) > 0
    assert profit > 0
