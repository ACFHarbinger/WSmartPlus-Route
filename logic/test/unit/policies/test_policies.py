"""Tests for policy implementations and solver engines."""

from typing import Callable, cast
from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies import run_hgs
from logic.src.policies.adaptive_large_neighborhood_search.policy_alns import ALNSPolicy
from logic.src.policies.base import PolicyRegistry
from logic.src.policies.capacitated_vehicle_routing_problem.cvrp import find_routes

# from logic.src.policies.branch_price_cut.policy_bcp import run_bcp, BCPPolicy
from logic.src.policies.hybrid_genetic_search.policy_hgs import HGSPolicy
from logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp import (
    ILSRVNDSPPolicy,
)
from logic.src.policies.travelling_salesman_problem import tsp


class MockBins:
    """Mock bins helper for policy tests."""
    def __init__(self, n=5):
        self.n = n
        self.c = np.zeros(n)
        self.means = np.ones(n) * 10.0
        self.std = np.ones(n) * 1.0
        self.collectlevl = 90.0


@pytest.fixture
def mock_policy_data():
    """Mock data for policy adapter tests."""
    n_bins = 5
    bins = MockBins(n_bins)
    bins.c[0] = 95.0

    dist_matrix = np.ones((n_bins + 1, n_bins + 1))
    np.fill_diagonal(dist_matrix, 0)

    return {
        "bins": bins,
        "distance_matrix": dist_matrix.tolist(),
        "waste_type": "plastic",
        "area": "test_area",
        "config": {},
        "distancesC": np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32),
        "must_go": [1],  # Pre-selected bin index 1 (ID 1)
    }


class TestPolicyAdapters:
    """
    Unit tests for standardized policy adapters and registry integration.
    """

    @pytest.fixture(autouse=True)
    def mock_loader(self):
        with patch("logic.src.pipeline.simulations.repository.load_area_and_waste_type_params") as mock_load:
            # Q, R, B, C, V
            mock_load.return_value = (100.0, 1.0, 1.0, 1.0, 1100.0)
            yield mock_load

    @pytest.mark.unit
    def test_alns_adapter(self, mock_policy_data):
        with patch("logic.src.policies.adaptive_large_neighborhood_search.policy_alns.run_alns") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, PolicyRegistry.get("alns"))()
            assert isinstance(policy, ALNSPolicy)
            tour, cost, extra = policy.execute(policy="alns_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    # @pytest.mark.unit
    # def test_bcp_adapter(self, mock_policy_data):
    #     with patch("logic.src.policies.branch_price_cut.policy_bcp.run_bcp") as mock_run:
    #         mock_run.return_value = ([[1]], 10.0)
    #         policy = cast(Callable, PolicyRegistry.get("bcp"))()
    #         assert isinstance(policy, BCPPolicy)
    #         tour, cost, extra = policy.execute(policy="bcp_1.0", **mock_policy_data)
    #         assert tour == [0, 1, 0]
    #         assert mock_run.called

    @pytest.mark.unit
    def test_ils_rvnd_sp_adapter(self, mock_policy_data):
        with patch("logic.src.policies.iterated_local_search_randomized_variable_neighborhood_descent_set_partitioning.policy_ils_rvnd_sp.ILSRVNDSPSolver") as mock_solver_cls:
            mock_solver_instance = mock_solver_cls.return_value
            mock_solver_instance.solve.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, PolicyRegistry.get("ils_rvnd_sp"))()
            assert isinstance(policy, ILSRVNDSPPolicy)
            tour, cost, extra = policy.execute(policy="ils_rvnd_sp_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_solver_instance.solve.called

    @pytest.mark.unit
    def test_hgs_adapter(self, mock_policy_data):
        with patch("logic.src.policies.hybrid_genetic_search.policy_hgs.run_hgs") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = cast(Callable, PolicyRegistry.get("hgs"))()
            assert isinstance(policy, HGSPolicy)
            tour, cost, extra = policy.execute(policy="hgs_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called


class TestAdvancedSolverEngines:
    """
    Unit tests for core solver engines (HGS, ALNS, BCP).
    """

    @pytest.mark.unit
    def test_hgs_custom(self, hgs_inputs):
        """Tests the custom HGS engine integration."""
        (
            dist_matrix,
            waste,
            capacity,
            R,
            C,
            global_must_go,
            local_to_global,
            vrpp_tour_global,
        ) = hgs_inputs
        values = {"hgs_engine": "custom", "time_limit": 1}
        routes, fitness, cost = run_hgs(
            dist_matrix,
            waste,
            capacity,
            R,
            C,
            values,
            global_must_go,
            local_to_global,
            vrpp_tour_global,
        )
        assert isinstance(routes, list)
        assert len(routes) > 0
        for r in routes:
            assert isinstance(r, list)

    @pytest.mark.unit
    def test_hgs_pyvrp(self, hgs_inputs):
        """Tests the PyVRP HGS engine integration."""
        (
            dist_matrix,
            waste,
            capacity,
            R,
            C,
            global_must_go,
            local_to_global,
            vrpp_tour_global,
        ) = hgs_inputs
        values = {"hgs_engine": "pyvrp", "time_limit": 1}
        routes, fitness, cost = run_hgs(
            dist_matrix,
            waste,
            capacity,
            R,
            C,
            values,
            global_must_go,
            local_to_global,
            vrpp_tour_global,
        )
        assert isinstance(routes, list)



    # @pytest.mark.unit
    # def test_bcp_variant_gurobi(self, mocker):
    #     \"\"\"Test Gurobi engine for BCP.\"\"\"
    #     from gurobipy import GRB
    #
    #     mock_model = MagicMock()
    #     ...


class TestMultiVehiclePolicies:
    """Tests for multi-vehicle routing policies."""

    @pytest.mark.unit
    def test_find_routes_basic(self):
        """Test basic multi-vehicle routing logic."""
        dist_matrix = np.array([
            [0, 10, 20, 10],
            [10, 0, 10, 20],
            [20, 10, 0, 10],
            [10, 20, 10, 0],
        ], dtype=np.int32)
        wastes = np.array([1, 1, 1])
        max_capacity = 5
        to_collect = np.array([1, 2, 3], dtype=np.int32)
        n_vehicles = 1
        depot = 0
        tour = find_routes(dist_matrix, wastes, max_capacity, to_collect, n_vehicles, depot=depot)
        assert isinstance(tour, list)
        assert set(tour) == {0, 1, 2, 3}


class TestSingleVehiclePolicies:
    """Test suite for single-vehicle routing policies and heuristics."""

    @pytest.mark.unit
    def test_find_route(self):
        """Test finding a route using TSP heuristic."""
        test_C = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]])
        test_to_collect = [1, 2]
        with patch("logic.src.policies.travelling_salesman_problem.tsp.fast_tsp.find_tour", return_value=[0, 1, 2]):
            res_route = tsp.find_route(test_C, test_to_collect)
            assert res_route == [0, 1, 2, 0]

    @pytest.mark.unit
    def test_get_route_cost(self):
        """Test calculation of total cost for a given route."""
        test_C = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]])
        test_tour = [0, 1, 2, 0]
        res_cost = tsp.get_route_cost(test_C, test_tour)
        assert abs(float(res_cost) - 35.0) < 1e-6
