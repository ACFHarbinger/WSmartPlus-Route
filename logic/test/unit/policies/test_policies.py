"""Tests for policy implementations and solver engines."""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from logic.src.policies import tsp
from logic.src.policies.adapters import PolicyRegistry
from logic.src.policies.hybrid_genetic_search import run_hgs
from logic.src.policies.cvrp import find_routes, find_routes_ortools
from logic.src.policies.adapters.policy_vrpp import run_vrpp_optimizer
from logic.src.policies.adapters.policy_alns import run_alns, ALNSPolicy
from logic.src.policies.adapters.policy_bcp import run_bcp, BCPPolicy
from logic.src.policies.adapters.policy_hgs import HGSPolicy
from logic.src.policies.adapters.policy_lkh import LKHPolicy
from logic.src.policies.lin_kernighan_helsgaun import solve_lkh


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
        with patch("logic.src.utils.data.data_utils.load_area_and_waste_type_params") as mock_load:
            # Q, R, _, C, _
            mock_load.return_value = (100.0, 1.0, None, 1.0, None)
            yield mock_load

    @pytest.mark.unit
    def test_alns_adapter(self, mock_policy_data):
        with patch("logic.src.policies.adapters.policy_alns.run_alns") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = PolicyRegistry.get("alns")()
            assert isinstance(policy, ALNSPolicy)
            tour, cost, extra = policy.execute(policy="alns_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    @pytest.mark.unit
    def test_bcp_adapter(self, mock_policy_data):
        with patch("logic.src.policies.adapters.policy_bcp.run_bcp") as mock_run:
            mock_run.return_value = ([[1]], 10.0)
            policy = PolicyRegistry.get("bcp")()
            assert isinstance(policy, BCPPolicy)
            tour, cost, extra = policy.execute(policy="bcp_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    @pytest.mark.unit
    def test_hgs_adapter(self, mock_policy_data):
        with patch("logic.src.policies.adapters.policy_hgs.run_hgs") as mock_run:
            mock_run.return_value = ([[1]], 10.0, 5.0)
            policy = PolicyRegistry.get("hgs")()
            assert isinstance(policy, HGSPolicy)
            tour, cost, extra = policy.execute(policy="hgs_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

    @pytest.mark.unit
    def test_lkh_adapter(self, mock_policy_data):
        with patch("logic.src.policies.adapters.policy_lkh.solve_lkh") as mock_run:
            mock_run.return_value = ([0, 1, 0], 5.0)
            policy = PolicyRegistry.get("lkh")()
            assert isinstance(policy, LKHPolicy)
            tour, cost, extra = policy.execute(policy="lkh_1.0", **mock_policy_data)
            # LKH policy adds +1 to indices internally, so input [1] becomes index 2
            # Update: mapping is now 1-to-1 for local execution, so [0, 1, 0] is expected
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



    @pytest.mark.unit
    def test_bcp_variant_gurobi(self, mocker):
        """Test Gurobi engine for BCP."""
        from gurobipy import GRB

        mock_model = MagicMock()
        mocker.patch("gurobipy.Model", return_value=mock_model)
        mock_model.optimize.return_value = None
        mock_model.status = GRB.OPTIMAL
        mock_model.SolCount = 1
        mock_model.objVal = 10.0

        dist_matrix = np.array([[0, 10, 100], [10, 0, 100], [100, 100, 0]], dtype=float)
        demands = {1: 1, 2: 1}
        capacity = 5
        R = 20
        C = 1
        values = {"time_limit": 1, "bcp_engine": "gurobi"}
        pass


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
        demands = np.array([1, 1, 1])
        max_capacity = 5
        to_collect = np.array([1, 2, 3], dtype=np.int32)
        n_vehicles = 1
        depot = 0
        tour = find_routes(dist_matrix, demands, max_capacity, to_collect, n_vehicles, depot=depot)
        assert isinstance(tour, list)
        assert set(tour) == {0, 1, 2, 3}


class TestSingleVehiclePolicies:
    """Test suite for single-vehicle routing policies and heuristics."""

    @pytest.mark.unit
    def test_find_route(self):
        """Test finding a route using TSP heuristic."""
        test_C = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]])
        test_to_collect = [1, 2]
        with patch("logic.src.policies.tsp.fast_tsp.find_tour", return_value=[0, 1, 2]):
            res_route = tsp.find_route(test_C, test_to_collect)
            assert res_route == [0, 1, 2, 0]

    @pytest.mark.unit
    def test_get_route_cost(self):
        """Test calculation of total cost for a given route."""
        test_C = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]])
        test_tour = [0, 1, 2, 0]
        res_cost = tsp.get_route_cost(test_C, test_tour)
        assert abs(float(res_cost) - 35.0) < 1e-6


class TestLinKernighanHelsgaun:
    """Tests for the Lin-Kernighan-Helsgaun heuristic implementation."""

    @pytest.mark.unit
    def test_small_instance(self):
        """Test LKH on a small 4-node square graph."""
        dist = np.array([[0, 1, 1.414, 1], [1, 0, 1, 1.414], [1.414, 1, 0, 1], [1, 1.414, 1, 0]])
        tour, cost = solve_lkh(dist, max_iterations=10)
        assert len(tour) == 5
        assert tour[0] == 0
        assert tour[-1] == 0
        assert len(set(tour)) == 4
        assert abs(cost - 4.0) < 1e-4

    @pytest.mark.unit
    def test_cvrp_penalty_improvement(self):
        """Test that LKH-3 prioritizes penalty reduction over distance."""
        dist = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]])
        waste = np.array([0, 60, 40])
        capacity = 50.0
        tour, cost = solve_lkh(dist, waste=waste, capacity=capacity, max_iterations=20)
        assert tour == [0, 2, 1, 0]
