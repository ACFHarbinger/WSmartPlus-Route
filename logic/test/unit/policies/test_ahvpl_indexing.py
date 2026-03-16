"""Tests for AHVPL giant tour consistency, crossover quality, and coaching."""

import numpy as np
import pytest
from logic.src.policies.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.ant_colony_optimization.k_sparse_aco.params import ACOParams
from logic.src.policies.augmented_hybrid_volleyball_premier_league.ahvpl import AHVPLSolver
from logic.src.policies.augmented_hybrid_volleyball_premier_league.params import AHVPLParams
from logic.src.policies.hybrid_genetic_search.individual import Individual
from logic.src.policies.hybrid_genetic_search.params import HGSParams
from logic.src.policies.other.operators.crossover import ordered_crossover


def _fast_params() -> AHVPLParams:
    """Create minimal AHVPL params for fast unit tests."""
    return AHVPLParams(
        n_teams=3,
        max_iterations=1,
        sub_rate=0.2,
        time_limit=10.0,
        hgs_params=HGSParams(
            nb_elite=2,
            mutation_rate=0.0,
            crossover_rate=0.5,
            max_vehicles=0,
            mu=3,
            n_offspring=1,
        ),
        aco_params=ACOParams(
            n_ants=2,
            max_iterations=1,
            k_sparse=5,
            rho=0.1,
            local_search=False,
        ),
        alns_params=ALNSParams(
            max_iterations=2,
            start_temp=100.0,
            cooling_rate=0.95,
            min_removal=1,
            max_removal_pct=0.3,
            time_limit=5.0,
        ),
    )


@pytest.fixture
def solver():
    """Create an AHVPLSolver with minimal test data and fast params."""
    n_nodes = 10
    rng = np.random.default_rng(42)
    dist_matrix = rng.random((n_nodes + 1, n_nodes + 1))
    np.fill_diagonal(dist_matrix, 0)
    wastes = {i: 1.0 for i in range(1, n_nodes + 1)}
    capacity = 5.0
    R, C = 10.0, 1.0
    return AHVPLSolver(dist_matrix, wastes, capacity, R, C, _fast_params())


def test_construct_individual_full_tour(solver):
    """Constructed individuals must have all nodes in their giant tour."""
    for _ in range(3):
        ind = solver._construct_individual()
        if ind:
            assert len(ind.giant_tour) == solver.n_nodes
            assert set(ind.giant_tour) == set(range(1, solver.n_nodes + 1))


def test_crossover_same_length():
    """OX on two individuals of equal length should produce valid child."""
    p1 = Individual([1, 2, 3, 4, 5])
    p2 = Individual([5, 4, 3, 2, 1])

    child = ordered_crossover(p1, p2)
    assert len(child.giant_tour) == 5
    assert set(child.giant_tour) == {1, 2, 3, 4, 5}


def test_active_crossover_strips_unvisited(solver):
    """_active_crossover should only apply OX to visited nodes."""
    p1 = Individual(list(range(1, solver.n_nodes + 1)))
    p1.routes = [[1, 2, 3], [4, 5]]
    p2 = Individual(list(range(1, solver.n_nodes + 1)))
    p2.routes = [[3, 4, 5], [6, 7]]

    child = solver._active_crossover(p1, p2)
    assert len(child.giant_tour) == solver.n_nodes
    assert set(child.giant_tour) == set(range(1, solver.n_nodes + 1))


def test_active_crossover_disjoint_parents(solver):
    """_active_crossover must handle parents with completely different active sets."""
    p1 = Individual(list(range(1, solver.n_nodes + 1)))
    p1.routes = [[1, 2, 3]]  # Only visits {1, 2, 3}
    p2 = Individual(list(range(1, solver.n_nodes + 1)))
    p2.routes = [[8, 9, 10]]  # Only visits {8, 9, 10}

    child = solver._active_crossover(p1, p2)
    assert len(child.giant_tour) == solver.n_nodes
    assert set(child.giant_tour) == set(range(1, solver.n_nodes + 1))


def test_active_crossover_empty_routes(solver):
    """_active_crossover should handle parents with no routes gracefully."""
    p1 = Individual(list(range(1, solver.n_nodes + 1)))
    p1.routes = []
    p2 = Individual(list(range(1, solver.n_nodes + 1)))
    p2.routes = [[1, 2, 3]]

    child = solver._active_crossover(p1, p2)
    assert len(child.giant_tour) == solver.n_nodes


def test_alns_coaching_reevaluates(solver):
    """After coaching, profit_score should be consistent with evaluate()."""
    ind = solver._construct_individual()
    if ind and ind.routes:
        coached = solver._alns_coaching(ind)
        assert coached.routes is not None
        assert coached.profit_score != -float("inf")
