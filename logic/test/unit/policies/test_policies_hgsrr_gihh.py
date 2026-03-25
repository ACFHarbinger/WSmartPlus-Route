"""
Tests for HGS-RR and GIHH policies.

This module tests the Hybrid Genetic Search with Ruin-and-Recreate (HGS-RR)
and the Hyper-Heuristic with Two Guidance Indicators (GIHH) policies.
"""

import numpy as np
import pytest
from logic.src.policies.hybrid_genetic_search_ruin_and_recreate import HGSRRSolver, HGSRRParams
from logic.src.policies.guided_indicators_hyper_heuristic import GIHHSolver, GIHHParams
from logic.src.policies.base import PolicyFactory


class TestHGSRRPolicy:
    """Test suite for HGS-RR policy."""

    @pytest.fixture
    def simple_instance(self):
        """Create a simple test instance."""
        # 5 nodes + depot
        dist_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 1.5, 2.5, 3.5, 4.5],
            [2.0, 1.5, 0.0, 1.0, 2.0, 3.0],
            [3.0, 2.5, 1.0, 0.0, 1.0, 2.0],
            [4.0, 3.5, 2.0, 1.0, 0.0, 1.0],
            [5.0, 4.5, 3.0, 2.0, 1.0, 0.0],
        ])
        wastes = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.3, 5: 0.4}
        capacity = 1.0
        R = 10.0  # Revenue per unit
        C = 1.0   # Cost per distance
        return dist_matrix, wastes, capacity, R, C

    def test_hgs_rr_basic_execution(self, simple_instance):
        """Test that HGS-RR executes without errors."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = HGSRRParams(time_limit=1.0, population_size=10, n_generations=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert isinstance(routes, list)
        assert isinstance(profit, float)
        assert isinstance(cost, float)

    def test_hgs_rr_respects_capacity(self, simple_instance):
        """Test that HGS-RR respects capacity constraints."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = HGSRRParams(time_limit=1.0, population_size=10, n_generations=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        # Check capacity for each route
        for route in routes:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity, f"Route {route} exceeds capacity"

    def test_hgs_rr_mandatory_nodes(self, simple_instance):
        """Test that HGS-RR visits mandatory nodes."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        mandatory_nodes = [1, 3]  # Local indices
        params = HGSRRParams(time_limit=1.0, population_size=10, n_generations=20, mutation_rate=0.3, seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
        routes, profit, cost = solver.solve()

        # Check that mandatory nodes are visited
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route)

        for node in mandatory_nodes:
            assert node in visited_nodes, f"Mandatory node {node} not visited"

    def test_hgs_rr_empty_instance(self):
        """Test HGS-RR with empty instance."""
        dist_matrix = np.array([[0.0]])
        wastes = {}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = HGSRRParams(seed=42)

        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert routes == []
        assert profit == 0.0
        assert cost == 0.0

    def test_hgs_rr_single_node(self):
        """Test HGS-RR with single node."""
        dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        wastes = {1: 0.5}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = HGSRRParams(seed=42)
        solver = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert len(routes) == 1
        assert routes[0] == [1]
        assert profit > 0
        assert cost > 0

    def test_hgs_rr_policy_adapter(self, simple_instance):
        """Test HGS-RR through PolicyFactory."""
        dist_matrix, wastes, capacity, R, C = simple_instance

        config = {
            "hgs_rr": {
                "time_limit": 1.0,
                "population_size": 10,
                "n_generations": 20,
                "seed": 42,
            }
        }

        policy = PolicyFactory.get_adapter("hgs_rr", config=config)
        assert policy is not None


class TestGIHHPolicy:
    """Test suite for GIHH policy."""

    @pytest.fixture
    def simple_instance(self):
        """Create a simple test instance."""
        # 5 nodes + depot
        dist_matrix = np.array([
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 0.0, 1.5, 2.5, 3.5, 4.5],
            [2.0, 1.5, 0.0, 1.0, 2.0, 3.0],
            [3.0, 2.5, 1.0, 0.0, 1.0, 2.0],
            [4.0, 3.5, 2.0, 1.0, 0.0, 1.0],
            [5.0, 4.5, 3.0, 2.0, 1.0, 0.0],
        ])
        wastes = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.3, 5: 0.4}
        capacity = 1.0
        R = 10.0  # Revenue per unit
        C = 1.0   # Cost per distance
        return dist_matrix, wastes, capacity, R, C

    def test_gihh_basic_execution(self, simple_instance):
        """Test that GIHH executes without errors."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHParams(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert isinstance(routes, list)
        assert isinstance(profit, float)
        assert isinstance(cost, float)

    def test_gihh_respects_capacity(self, simple_instance):
        """Test that GIHH respects capacity constraints."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHParams(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        # Check capacity for each route
        for route in routes:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity, f"Route {route} exceeds capacity"

    def test_gihh_mandatory_nodes(self, simple_instance):
        """Test that GIHH visits mandatory nodes."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        mandatory_nodes = [1, 3]  # Local indices
        params = GIHHParams(time_limit=1.0, max_iterations=5, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
        routes, profit, cost = solver.solve()

        # Check that mandatory nodes are visited
        visited_nodes = set()
        for route in routes:
            visited_nodes.update(route)

        for node in mandatory_nodes:
            assert node in visited_nodes, f"Mandatory node {node} not visited"

    def test_gihh_empty_instance(self):
        """Test GIHH with empty instance."""
        dist_matrix = np.array([[0.0]])
        wastes = {}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = GIHHParams(seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert routes == []
        assert profit == 0.0
        assert cost == 0.0

    def test_gihh_single_node(self):
        """Test GIHH with single node."""
        dist_matrix = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        wastes = {1: 0.5}
        capacity = 1.0
        R = 10.0
        C = 1.0
        params = GIHHParams(seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert len(routes) == 1
        assert routes[0] == [1]
        assert profit > 0
        assert cost > 0

    def test_gihh_indicator_weights(self, simple_instance):
        """Test GIHH with custom indicator weights."""
        dist_matrix, wastes, capacity, R, C = simple_instance
        params = GIHHParams(time_limit=1.0, max_iterations=5, iri_weight=0.7, tbi_weight=0.3, seed=42)
        solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params, None)
        routes, profit, cost = solver.solve()

        assert isinstance(routes, list)
        assert isinstance(profit, float)

    def test_gihh_policy_adapter(self, simple_instance):
        """Test GIHH through PolicyFactory."""
        dist_matrix, wastes, capacity, R, C = simple_instance

        config = {
            "gihh": {
                "time_limit": 1.0,
                "max_iterations": 5,
                "seed": 42,
            }
        }

        policy = PolicyFactory.get_adapter("gihh", config=config)
        assert policy is not None


class TestPolicyComparison:
    """Compare HGS-RR and GIHH on the same instances."""

    @pytest.fixture
    def test_instance(self):
        """Create a test instance for comparison."""
        np.random.seed(42)
        n = 10
        dist_matrix = np.random.uniform(0.5, 5.0, (n + 1, n + 1))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Symmetric
        np.fill_diagonal(dist_matrix, 0.0)

        wastes = {i: np.random.uniform(0.1, 0.4) for i in range(1, n + 1)}
        capacity = 1.0
        R = 10.0
        C = 1.0

        return dist_matrix, wastes, capacity, R, C

    def test_both_policies_feasible(self, test_instance):
        """Test that both policies produce feasible solutions."""
        dist_matrix, wastes, capacity, R, C = test_instance

        params_hgs_rr = HGSRRParams(
            time_limit=2.0,
            population_size=20,
            n_generations=50,
            min_removal_pct=0.05,  # Reduce removal to avoid edge cases
            max_removal_pct=0.2,
            seed=42,
        )

        params_gihh = GIHHParams(
            time_limit=2.0,
            max_iterations=100,
            seed=42,
        )

        solver_hgs_rr = HGSRRSolver(dist_matrix, wastes, capacity, R, C, params_hgs_rr, None)
        routes_hgs_rr, profit_hgs_rr, cost_hgs_rr = solver_hgs_rr.solve()

        solver_gihh = GIHHSolver(dist_matrix, wastes, capacity, R, C, params_gihh, None)
        routes_gihh, profit_gihh, cost_gihh = solver_gihh.solve()

        # Both should produce valid solutions
        assert isinstance(routes_hgs_rr, list)
        assert isinstance(routes_gihh, list)

        # Check feasibility for HGS-RR
        for route in routes_hgs_rr:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity

        # Check feasibility for GIHH
        for route in routes_gihh:
            route_load = sum(wastes.get(node, 0.0) for node in route)
            assert route_load <= capacity
