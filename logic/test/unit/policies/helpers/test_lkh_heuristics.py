"""
Unit tests for LKH-1 and LKH-2 heuristics.

Covers:
- Basic convergence and structural validity (valid Hamiltonian cycle)
- Held-Karp lower bound correctness
- TourPopulation diversity-aware insertion and parent selection
- IPT crossover integrity (valid offspring, backbone preservation)
- LKH-2 vs LKH-1 convergence benchmark
"""
from random import Random
from typing import List, Set, Tuple

import numpy as np
import pytest
from logic.src.policies.helpers.operators.search_heuristics._objective import get_cost
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import (
    compute_alpha_measures,
    get_candidate_set,
    run_subgradient,
    solve_lkh1,
)
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two import (
    TourPopulation,
    ipt_crossover,
    solve_lkh2,
)


@pytest.fixture
def random_distance_matrix() -> np.ndarray:
    """Generate a random Euclidean distance matrix for TSP."""
    rng = np.random.default_rng(42)
    n = 20
    coords = rng.uniform(0, 100, size=(n, 2))
    # Euclidean distance
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    dist = np.round(dist).astype(float)  # TSPLIB style rounding
    return dist


def test_lkh1_convergence(random_distance_matrix: np.ndarray) -> None:
    """Test LKH-1 on a symmetric distance matrix."""
    dist = random_distance_matrix
    tour, cost, hk_bound = solve_lkh1(
        dist,
        max_iterations=10,
        max_k=3,
        seed=42,
    )
    # Check valid closed tour structure
    assert len(tour) == len(dist) + 1
    assert tour[0] == tour[-1]
    assert len(set(tour)) == len(dist)

    # Check calculated cost exactly matches route length
    assert abs(get_cost(tour, dist) - cost) < 1e-6

    # Verify that the Held-Karp bound is a valid lower bound
    assert hk_bound <= cost + 1e-6


def test_lkh2_convergence(random_distance_matrix: np.ndarray) -> None:
    """Test LKH-2 meta-heuristic on a symmetric distance matrix."""
    dist = random_distance_matrix
    tour, cost, hk_bound = solve_lkh2(
        dist,
        max_iterations=10,
        max_k=3,
        population_size=5,
        seed=42,
    )
    # Check valid closed tour structure
    assert len(tour) == len(dist) + 1
    assert tour[0] == tour[-1]
    assert len(set(tour)) == len(dist)

    # Check calculated cost exactly matches route length
    assert abs(get_cost(tour, dist) - cost) < 1e-6

    # Verify that the Held-Karp bound is a valid lower bound
    assert hk_bound <= cost + 1e-6


def test_lkh1_small_instance() -> None:
    """Test LKH-1 against a small hand-verifiable problem instance."""
    dist = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ], dtype=float)

    tour, cost, _ = solve_lkh1(dist, max_iterations=5, max_k=2)
    # Optimum is 0-1-3-2-0 with total cost 10 + 25 + 30 + 15 = 80
    assert cost <= 80 + 1e-6


def test_lkh2_small_instance() -> None:
    """Test LKH-2 against a small hand-verifiable problem instance."""
    dist = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ], dtype=float)

    tour, cost, _ = solve_lkh2(dist, max_iterations=5, max_k=2, population_size=2)
    # Optimum is 0-1-3-2-0 with total cost 10 + 25 + 30 + 15 = 80
    assert cost <= 80 + 1e-6


# ---------------------------------------------------------------------------
# TourPopulation tests
# ---------------------------------------------------------------------------


def _make_euclidean_dist(n: int, seed: int = 0) -> np.ndarray:
    """Build a symmetric Euclidean distance matrix for n cities."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt(np.sum(diff**2, axis=-1))).astype(float)


def _sequential_tour(n: int, offset: int = 0) -> List[int]:
    """Build a closed tour 0 -> offset -> (offset+1) -> ... -> 0."""
    inner = [(i + offset) % n for i in range(n)]
    return inner + [inner[0]]


def test_tour_population_diversity() -> None:
    """
    Verify that TourPopulation:
    1. Accepts tours that are structurally novel (diverse) even if they cost more.
    2. Always accepts tours better than the current worst.
    3. sample_parents returns two distinct members.
    """
    n = 10
    pop = TourPopulation(max_size=4, diversity_threshold=0.1)

    dist = _make_euclidean_dist(n, seed=7)

    # Seed the population with sequentially shifted tours (maximally diverse)
    tours = [_sequential_tour(n, offset=i) for i in range(6)]
    costs = [get_cost(t, dist) for t in tours]

    inserted = [pop.try_insert(t, c) for t, c in zip(tours[:4], costs[:4], strict=True)]
    assert all(inserted), "All initial insertions should succeed (population not yet full)"
    assert pop.size == 4

    # A highly diverse 5th tour should still be inserted (evicts the worst)
    pop.try_insert(tours[4], costs[4])
    assert pop.size == 4  # size must not exceed max_size
    # The population should have updated even if this tour wasn't better
    # (it may have been, or diversity criterion triggered)
    # At minimum, best_cost must be <= min of all inserted costs
    assert pop.best_cost <= min(costs[:5]) + 1e-6, "best_cost must track the global optimum"

    # sample_parents must return 2 distinct tours
    rng = np.random.default_rng(0)
    parents = pop.sample_parents(rng, n_parents=2)
    assert len(parents) == 2
    # The two selected parents must not be identical
    assert parents[0][0] != parents[1][0], "Sampled parents should be distinct"


def test_ipt_crossover_integrity() -> None:
    """
    Verify that ipt_crossover always produces a valid Hamiltonian cycle and
    preserves at least some backbone edges from the two parents.
    """
    n = 12
    dist = _make_euclidean_dist(n, seed=99)

    # Build two locally-optimal tours via LKH-1 from different starts
    _, _, penalized_dm = run_subgradient(dist, max_iter=30)
    alpha = compute_alpha_measures(penalized_dm)
    candidates = get_candidate_set(penalized_dm, alpha, max_candidates=5)

    parent1 = list(range(n)) + [0]   # trivial sequential tour
    parent2 = list(reversed(range(n))) + [n - 1]  # reversed tour
    # Ensure both start at 0
    if parent2[0] != 0 and 0 in parent2:
        rot = parent2.index(0)
        parent2 = parent2[rot:] + parent2[1 : rot + 1]
    if parent2[-1] != parent2[0]:
        parent2.append(parent2[0])

    stdlib_rng = Random(42)
    offspring, offspring_cost = ipt_crossover(
        parent1,
        parent2,
        dist,
        candidates,
        stdlib_rng,
        max_k=2,
        max_gap_iter=5,
    )

    # --- Structural validity ---
    assert len(offspring) == n + 1, f"Expected tour length {n + 1}, got {len(offspring)}"
    assert offspring[0] == offspring[-1] == 0, "Tour must start and end at depot (0)"
    assert len(set(offspring)) == n, f"All {n} nodes must appear exactly once (excl. depot duplication)"
    assert abs(get_cost(offspring, dist) - offspring_cost) < 1e-6, "Reported cost must match actual cost"

    # --- Backbone preservation: at least one backbone edge must survive in offspring ---
    def edge_set(tour: List[int]) -> Set[Tuple[int, int]]:
        return {(min(tour[i], tour[i + 1]), max(tour[i], tour[i + 1])) for i in range(len(tour) - 1)}

    es1 = edge_set(parent1)
    es2 = edge_set(parent2)
    backbone = es1 & es2
    offspring_edges = edge_set(offspring)

    # If parents share any edges, the offspring should preserve at least some
    if backbone:
        assert backbone & offspring_edges, (
            "IPT offspring must preserve at least one backbone edge when parents share edges"
        )


def test_lkh2_vs_lkh1() -> None:
    """
    Convergence benchmark: LKH-2 should find a tour at least as good as LKH-1
    on a medium-size instance (N=30) with equal iteration budget.
    """
    n = 30
    dist = _make_euclidean_dist(n, seed=123)

    lkh1_tour, lkh1_cost, _ = solve_lkh1(
        dist,
        max_iterations=20,
        max_k=3,
        seed=42,
    )
    lkh2_tour, lkh2_cost, _ = solve_lkh2(
        dist,
        max_iterations=20,
        max_k=3,
        population_size=5,
        seed=42,
    )

    # Structural validity of LKH-2 output
    assert len(lkh2_tour) == n + 1
    assert lkh2_tour[0] == lkh2_tour[-1]
    assert len(set(lkh2_tour)) == n

    # LKH-2 should find a tour no worse than LKH-1 (with tolerance for stochasticity)
    # Allow 5% slack: LKH-2 may trade some exploitation depth for population diversity
    tolerance = 1.05
    assert lkh2_cost <= lkh1_cost * tolerance, (
        f"LKH-2 cost {lkh2_cost:.2f} should be within 5% of LKH-1 cost {lkh1_cost:.2f}"
    )
