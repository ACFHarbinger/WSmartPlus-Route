"""
Comprehensive test suite for policy auxiliary modules.

This module consolidates tests for:
- Look-ahead auxiliary functions (swap, select, solutions, check, move, update)
- HGS (Hybrid Genetic Search) auxiliary functions (split, evolution, local_search)
- ALNS (Adaptive Large Neighborhood Search) auxiliary functions (destroy, repair)
"""

from unittest.mock import patch

from logic.src.policies.simulated_annealing_neighborhood_search.common.solution_initialization import (
    find_initial_solution, compute_initial_solution
)
from logic.src.policies.simulated_annealing_neighborhood_search.heuristics.sans import (
    improved_simulated_annealing,
)
import numpy as np
import pandas as pd
import pytest

# HGS and ALNS auxiliary imports
from logic.src.policies.operators import destroy_operators, repair_operators
from logic.src.policies.hybrid_genetic_search import evolution, types
from logic.src.policies.hybrid_genetic_search import split as split_module
from logic.src.policies import local_search

# Look-ahead auxiliary imports
from logic.src.policies.simulated_annealing_neighborhood_search.common.check import (
    check_bins_overflowing_feasibility,
    check_solution_admissibility,
)
from logic.src.policies.simulated_annealing_neighborhood_search.operators.move import (
    move_1_route,
    move_2_routes,
    move_n_2_routes_consecutive,
    move_n_2_routes_random,
    move_n_route_consecutive,
    move_n_route_random,
)
from logic.src.policies.simulated_annealing_neighborhood_search.select import (
    add_bin,
    add_n_bins_random,
    add_route_random,
    add_route_with_removed_bins_random,
    remove_bin,
    remove_n_bins_random,
)
from logic.src.policies.simulated_annealing_neighborhood_search.operators import (
    swap_1_route,
    swap_2_routes,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)

# ============================================================================
# Look-Ahead Auxiliary Tests
# ============================================================================


class TestLookAheadCheck:
    """Tests for look-ahead checking utilities."""

    def test_check_bins_overflowing_feasibility(self):
        """Test bin overflow feasibility checking."""
        data = pd.DataFrame({"#bin": [1, 2, 3], "Stock": [80, 90, 70], "Accum_Rate": [5, 10, 3]})
        routes = [[1, 2], [3]]

        # Actual signature: (data, routes_list, number_of_bins, perc_bins_can_overflow, E, B)
        result = check_bins_overflowing_feasibility(data, routes, 3, 0.5, 100.0, 1.0)
        assert isinstance(result, str)
        assert result in ["pass", "fail"]

    def test_check_solution_admissibility(self):
        """Test solution admissibility checking."""
        routes = [[0, 1, 2, 0], [0, 3, 0]]
        removed_bins = []

        # Actual signature: (routes_list, removed_bins, number_of_bins)
        result = check_solution_admissibility(routes, removed_bins, 3)
        assert isinstance(result, bool)


class TestLookAheadSelect:
    """Tests for look-ahead selection operations."""

    @pytest.fixture
    def sample_routes(self):
        """Fixture providing a sample set of routing results."""
        return [[0, 1, 2, 0], [0, 3, 4, 0]]

    def test_remove_bin(self, sample_routes):
        """Test bin removal from routes."""
        removed_bins = []
        cannot_remove = []
        bin_rem = remove_bin(sample_routes, removed_bins, cannot_remove)
        assert bin_rem is not None or len(sample_routes) == 0

    def test_add_bin(self):
        """Test bin addition to routes."""
        removed_bins = [7]
        routes = [[0, 1, 2, 0]]
        bin_add = add_bin(routes, removed_bins)
        assert bin_add == 7 or bin_add is None

    def test_remove_n_bins_random(self):
        """Test random removal of multiple bins."""
        routes = [[0, 1, 2, 3, 4, 5, 0]]
        removed_bins = []
        bins_cannot_removed = []

        # Actual signature: (routes_list, removed_bins, bins_cannot_removed)
        # Returns (bins_to_remove_random, chosen_n)
        result, n = remove_n_bins_random(routes, removed_bins, bins_cannot_removed)
        assert isinstance(result, list)
        assert isinstance(n, int)

    def test_add_n_bins_random(self):
        """Test random addition of multiple bins."""
        routes = [[0, 1, 2, 0]]
        removed_bins = [7, 8, 9]
        added, n = add_n_bins_random(routes, removed_bins)
        assert isinstance(added, list)

    def test_add_route_random(self):
        """Test random route addition."""
        routes = [[0, 1, 2, 0]]
        dist_matrix = np.ones((10, 10))
        np.fill_diagonal(dist_matrix, 0)

        add_route_random(routes, dist_matrix)
        assert len(routes) >= 1

    def test_add_route_with_removed_bins_random(self):
        """Test adding route with removed bins."""
        routes = [[0, 1, 2, 0]]
        removed_bins = [7, 8, 9]
        dist = np.ones((10, 10))
        np.fill_diagonal(dist, 0)

        n, used = add_route_with_removed_bins_random(routes, removed_bins, dist)
        assert isinstance(used, list)


class TestLookAheadMove:
    """Tests for look-ahead move operations."""

    @pytest.fixture
    def sample_routes(self):
        """Fixture providing a sample set of routing results for move operations."""
        return [[0, 1, 2, 3, 0], [0, 4, 5, 0]]

    def test_move_1_route(self, sample_routes):
        """Test single route move operation."""
        move_1_route(sample_routes)
        assert len(sample_routes) >= 1

    def test_move_2_routes(self, sample_routes):
        """Test two-route move operation."""
        move_2_routes(sample_routes)
        assert len(sample_routes) >= 1

    def test_move_n_route_random(self, sample_routes):
        """Test random n-route move."""
        with patch("random.sample") as mock_sample:

            def side_eff(pop, k):
                """Mock side effect for random sample."""
                return [pop[0]] if k == 1 else pop[:k]

            mock_sample.side_effect = side_eff

            move_n_route_random(sample_routes, n=2)
            assert isinstance(sample_routes, list)

    def test_move_n_route_consecutive(self, sample_routes):
        """Test consecutive n-route move."""
        with patch("random.sample") as mock_sample:
            mock_sample.side_effect = lambda pop, k: pop[:k]
            move_n_route_consecutive(sample_routes, n=2)
            assert isinstance(sample_routes, list)

    def test_move_n_2_routes_random(self, sample_routes):
        """Test random n-move between 2 routes."""
        # Actual signature: (routes_list) - no n parameter, randomly chosen internally
        move_n_2_routes_random(sample_routes)
        assert isinstance(sample_routes, list)

    def test_move_n_2_routes_consecutive(self, sample_routes):
        """Test consecutive n-move between 2 routes."""
        # Actual signature: (routes_list) - no n parameter, randomly chosen internally
        move_n_2_routes_consecutive(sample_routes)
        assert isinstance(sample_routes, list)


class TestLookAheadSwap:
    """Tests for look-ahead swap operations."""

    @pytest.fixture
    def sample_routes(self):
        """Fixture providing a sample set of routing results for swap operations."""
        return [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 0]]

    def test_swap_1_route(self, sample_routes):
        """Test single route swap operation."""
        swap_1_route(sample_routes)
        assert len(sample_routes) >= 1

    def test_swap_2_routes(self, sample_routes):
        """Test two-route swap operation."""
        swap_2_routes(sample_routes)
        assert len(sample_routes) >= 1

    def test_swap_n_route_random(self, sample_routes):
        """Test random n-swap within route."""

        def run_with_n(n: int) -> None:
            """Helper to run swap test with different n values."""
            routes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]

            with patch("random.sample") as mock_sample:

                def side_eff(pop, k):
                    """Mock side effect for random selection in swap."""
                    if len(pop) < k:
                        return pop
                    return pop[:k]

                mock_sample.side_effect = side_eff

                swap_n_route_random(routes, n=n)
                assert isinstance(routes, list)

        run_with_n(2)
        run_with_n(3)

    def test_swap_n_route_consecutive(self, sample_routes):
        """Test consecutive n-swap within route."""
        with patch("random.sample") as mock_sample:
            mock_sample.side_effect = lambda pop, k: pop[:k]
            swap_n_route_consecutive(sample_routes, n=2)
            assert isinstance(sample_routes, list)

    def test_swap_n_2_routes_random(self, sample_routes):
        """Test random n-swap between 2 routes."""
        # Actual signature: (routes_list) - no n parameter, randomly chosen internally
        swap_n_2_routes_random(sample_routes)
        assert isinstance(sample_routes, list)

    def test_swap_n_2_routes_consecutive(self, sample_routes):
        """Test consecutive n-swap between 2 routes."""
        # Actual signature: (routes_list) - no n parameter, randomly chosen internally
        swap_n_2_routes_consecutive(sample_routes)
        assert isinstance(sample_routes, list)


class TestLookAheadSolutions:
    """Tests for look-ahead solution generation."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing a sample dataframe of bin data."""
        return pd.DataFrame(
            {
                "#bin": [i for i in range(7)],
                "Weight": [10] * 7,
                "Reward": [1.0] * 7,
                "Stock": [50] * 7,
                "Accum_Rate": [1.0] * 7,
            }
        )

    @pytest.fixture
    def bins_coord_dict(self):
        """Fixture providing bin coordinates as a dictionary."""
        return {i: [0.0, 0.0] for i in range(7)}

    @pytest.fixture
    def bins_coord_df(self):
        """Fixture providing bin coordinates as a dataframe."""
        return pd.DataFrame({"Lat": [0.0] * 7, "Lng": [0.0] * 7}, index=[i for i in range(7)])

    @pytest.fixture
    def dist_matrix(self):
        """Fixture providing a sample distance matrix."""
        return np.ones((7, 7))

    def test_find_initial_solution(self, sample_data, bins_coord_df, dist_matrix):
        """Test initial solution finding."""
        res = find_initial_solution(sample_data, bins_coord_df, dist_matrix, 6, 2000.0, 0.5, 20.0)
        assert isinstance(res, list)

    def test_compute_initial_solution(self, sample_data, bins_coord_dict, dist_matrix):
        """Test initial solution computation."""
        id_to_index = {i: i for i in range(7)}
        res = compute_initial_solution(sample_data, bins_coord_dict, dist_matrix, 2000.0, id_to_index)
        assert isinstance(res, list)

    def test_simulated_annealing_smoke(self, sample_data, bins_coord_dict, dist_matrix):
        """Test simulated annealing optimization."""
        routes = [[1, 2, 3], [4, 5, 6]]
        id_to_index = {i: i for i in range(7)}
        res = improved_simulated_annealing(
            routes,
            dist_matrix,
            time_limit=0.1,
            verbose=False,
            id_to_index=id_to_index,
            data=sample_data,
            vehicle_capacity=1000.0,
        )
        assert isinstance(res, tuple)
        assert len(res) == 5


# ============================================================================
# ALNS Auxiliary Tests
# ============================================================================


class TestALNSAux:
    """Tests for ALNS auxiliary functions."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample ALNS problem data."""
        return {
            "dist_matrix": np.array([[0, 10, 10, 10], [10, 0, 5, 5], [10, 5, 0, 5], [10, 5, 5, 0]]),
            "demands": {1: 10, 2: 10, 3: 10},
            "values": {1: 100, 2: 100, 3: 100},
            "capacity": 20.0,
            "R": 1.0,
            "C": 0.5,
        }

    def test_random_removal(self, sample_data):
        """Test random removal destroy operator."""
        solution = [[1, 2], [3]]
        routes, removed = destroy_operators.random_removal(solution, 2)
        assert len(removed) == 2

    def test_worst_removal(self, sample_data):
        """Test worst removal destroy operator."""
        solution = [[1, 2], [3]]
        routes, removed = destroy_operators.worst_removal(solution, 2, sample_data["dist_matrix"])
        assert len(removed) == 2

    def test_greedy_insertion(self, sample_data):
        """Test greedy insertion repair operator."""
        solution = [[1]]
        removed = [2, 3]
        res = repair_operators.greedy_insertion(
            solution,
            removed,
            sample_data["dist_matrix"],
            sample_data["demands"],
            sample_data["capacity"],
        )
        assert isinstance(res, list)

    def test_regret_2_insertion(self, sample_data):
        """Test regret-2 insertion repair operator."""
        solution = [[1]]
        removed = [2, 3]
        res = repair_operators.regret_2_insertion(
            solution,
            removed,
            sample_data["dist_matrix"],
            sample_data["demands"],
            sample_data["capacity"],
        )
        assert isinstance(res, list)


# ============================================================================
# HGS Auxiliary Tests
# ============================================================================


class TestHGSAux:
    """Tests for HGS auxiliary functions."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample HGS problem data."""
        return {
            "dist_matrix": np.array([[0, 10, 10, 10], [10, 0, 5, 5], [10, 5, 0, 5], [10, 5, 5, 0]]),
            "demands": {1: 10, 2: 10, 3: 10},
            "capacity": 20.0,
            "R": 1.0,
            "C": 0.5,
        }

    def test_split_algorithm(self, sample_data):
        """Test split algorithm for giant tour decomposition."""
        split = split_module.LinearSplit(
            sample_data["dist_matrix"],
            sample_data["demands"],
            sample_data["capacity"],
            sample_data["R"],
            sample_data["C"],
        )
        giant_tour = [1, 2, 3]
        routes, profit = split.split(giant_tour)
        assert isinstance(routes, list)
        assert profit >= 0

    def test_hgs_evolution_ox(self):
        """Test ordered crossover evolution operator."""
        p1 = types.Individual([1, 2, 3, 4, 5])
        p2 = types.Individual([5, 4, 3, 2, 1])
        child = evolution.ordered_crossover(p1, p2)
        assert len(child.giant_tour) == 5
        assert set(child.giant_tour) == {1, 2, 3, 4, 5}

    def test_hgs_biased_fitness(self):
        """Test biased fitness calculation."""
        ind1 = types.Individual([1, 2, 3])
        ind1.profit_score = 100
        ind2 = types.Individual([3, 2, 1])
        ind2.profit_score = 50
        pop = [ind1, ind2]
        evolution.update_biased_fitness(pop, nb_elite=1)
        assert hasattr(ind1, "fitness")
        assert ind1.rank_profit == 1

    def test_hgs_local_search(self, sample_data):
        """Test local search optimization."""
        params = types.HGSParams(time_limit=1.0)
        ls = local_search.LocalSearch(
            sample_data["dist_matrix"],
            sample_data["demands"],
            sample_data["capacity"],
            sample_data["R"],
            sample_data["C"],
            params,
        )
        ind = types.Individual([1, 2, 3])
        ind.routes = [[1, 2], [3]]
        improved = ls.optimize(ind)
        assert isinstance(improved, types.Individual)
