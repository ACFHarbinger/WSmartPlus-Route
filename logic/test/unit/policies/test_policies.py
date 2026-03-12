"""Tests for policy implementations and solver engines."""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from typing import cast, Callable

from logic.src.policies.travelling_salesman_problem import tsp
from logic.src.policies.capacitated_vehicle_routing_problem import cvrp
from logic.src.policies.base import PolicyRegistry
from logic.src.policies import run_hgs
from logic.src.policies.capacitated_vehicle_routing_problem.cvrp import find_routes, find_routes_ortools
from logic.src.policies.vehicle_routing_problem_with_profits.policy_vrpp import run_vrpp_optimizer
from logic.src.policies.adaptive_large_neighborhood_search.policy_alns import run_alns, ALNSPolicy
from logic.src.policies.branch_cut_and_price.policy_bcp import run_bcp, BCPPolicy
from logic.src.policies.hybrid_genetic_search.policy_hgs import HGSPolicy


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

    @pytest.mark.unit
    def test_bcp_adapter(self, mock_policy_data):
        with patch("logic.src.policies.branch_cut_and_price.policy_bcp.run_bcp") as mock_run:
            mock_run.return_value = ([[1]], 10.0)
            policy = cast(Callable, PolicyRegistry.get("bcp"))()
            assert isinstance(policy, BCPPolicy)
            tour, cost, extra = policy.execute(policy="bcp_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

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
        wastes = {1: 1, 2: 1}
        capacity = 5
        R = 20
        C = 1
        values = {"time_limit": 1, "bcp_engine": "gurobi"}
        # Mock decision variables forExtraction logic
        mock_vars = {}

        def mock_add_var(vtype=None, name="", **kwargs):
            m_var = MagicMock()
            # Support Gurobi operator overloading for expressions
            for op in ["__add__", "__sub__", "__mul__", "__rmul__", "__ge__", "__le__", "__eq__"]:
                setattr(m_var, op, MagicMock(return_value=m_var))

            if name == "x_0_1" or name == "x_1_0":
                m_var.X = 1.0
            else:
                m_var.X = 0.0
            mock_vars[name] = m_var
            return m_var

        mock_model.addVar.side_effect = mock_add_var

        routes, cost = run_bcp(dist_matrix, wastes, capacity, R, C, values)

        # Assertions
        assert routes == [[1]]
        assert cost == 10.0
        assert mock_model.optimize.called
        # Check if mandatory visit constraint was added if we provided must_go_indices
        # In this call we didn't, so let's verify a call with must_go_indices as well
        must_go_values = {"time_limit": 1, "bcp_engine": "gurobi"}
        run_bcp(dist_matrix, wastes, capacity, R, C, must_go_values, must_go_indices={1})
        # Verify that addConstr was called for the must_visit constraint
        # The name in gurobi_engine.py is f"must_visit_{i}"
        mock_model.addConstr.assert_any_call(mock_vars["y_1"] == 1, name="must_visit_1")


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
