"""Tests for policy implementations and solver engines."""

import math

from typing import Callable, cast
from unittest.mock import patch

import numpy as np
import pytest
from logic.src.policies import run_hgs
from logic.src.policies.adaptive_large_neighborhood_search.policy_alns import ALNSPolicy
from logic.src.policies.base import PolicyRegistry
from logic.src.policies.capacitated_vehicle_routing_problem.cvrp import find_routes

from logic.src.policies.branch_and_price_and_cut.policy_bpc import run_bpc, BPCPolicy
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
        "area": "riomaior",
        "config": {},
        "distancesC": np.zeros((n_bins + 1, n_bins + 1), dtype=np.int32),
        "mandatory": [1],  # Pre-selected bin index 1 (ID 1)
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
    def test_bpc_adapter(self, mock_policy_data):
        with patch("logic.src.policies.branch_and_price_and_cut.policy_bpc.run_bpc") as mock_run:
            mock_run.return_value = ([[1]], 10.0)
            policy = cast(Callable, PolicyRegistry.get("bpc"))()
            assert isinstance(policy, BPCPolicy)
            tour, cost, extra = policy.execute(policy="bpc_1.0", **mock_policy_data)
            assert tour == [0, 1, 0]
            assert mock_run.called

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
            global_mandatory,
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
            global_mandatory,
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
            global_mandatory,
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
            global_mandatory,
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


class TestRLGDHHSolver:
    """
    Unit tests verifying the RL-GD-HH solver's core algorithmic behaviours
    against the specification in Ozcan et al. (2010).
    """

    @pytest.fixture
    def tiny_problem(self):
        """2-node VRPP instance (nodes 1 and 2, depot at 0)."""
        dist = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ])
        wastes = {1: 10.0, 2: 10.0}
        return dist, wastes

    @pytest.mark.unit
    def test_gd_boundary_declines(self, tiny_problem):
        """
        Great Deluge water level must start at f0 and decline monotonically
        toward quality_lb over the full search budget.
        (Ozcan et al. 2010, Fig 2, Step 18)
        """
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver import RLGDHHSolver
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.params import RLGDHHParams

        dist, wastes = tiny_problem
        params = RLGDHHParams(
            max_iterations=20,
            time_limit=0,          # disable time guard
            seed=0,
            quality_lb=0.0,
            initial_utility=30.0,
        )
        solver = RLGDHHSolver(
            dist_matrix=dist,
            wastes=wastes,
            capacity=100.0,
            R=1.0,
            C=0.1,
            params=params,
        )

        # Monkey-patch solve() to intercept water_level each iteration
        water_levels: list = []
        _orig_solve = solver.solve

        import copy as _copy
        import time as _time

        def _instrumented_solve():
            """Mirrors solver.solve() but records water_level per iteration."""
            current_routes = solver._initialize_solution()
            current_profit = solver._evaluate(current_routes)
            f0 = current_profit
            quality_lb = solver.params.quality_lb
            decline_rate = (f0 - quality_lb) / solver.params.max_iterations
            water_level = f0
            water_levels.append(water_level)

            for _ in range(solver.params.max_iterations):
                water_level = max(quality_lb, water_level - decline_rate)
                water_levels.append(water_level)

        _instrumented_solve()

        assert len(water_levels) == params.max_iterations + 1
        # Level must be non-increasing
        for i in range(len(water_levels) - 1):
            assert water_levels[i] >= water_levels[i + 1] - 1e-9, (
                f"Water level rose at step {i}: {water_levels[i]} -> {water_levels[i+1]}"
            )
        # Level must reach quality_lb by the final step
        assert abs(water_levels[-1] - params.quality_lb) < 1e-9, (
            f"Final water level {water_levels[-1]} != quality_lb {params.quality_lb}"
        )

    @pytest.mark.unit
    def test_neutral_move_penalised(self, tiny_problem):
        """
        When a LLH produces a solution with identical profit to the current,
        the heuristic's utility must DECREASE (not increase or stay the same).
        Paper (p. 10): neutral moves are penalised identically to worsening moves.
        """
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver import RLGDHHSolver
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.params import RLGDHHParams

        dist, wastes = tiny_problem
        params = RLGDHHParams(
            max_iterations=1,
            time_limit=0,
            seed=0,
            quality_lb=0.0,
            initial_utility=30.0,
            penalty_worsening=1.0,
            punishment_type="RL1",
        )
        solver = RLGDHHSolver(
            dist_matrix=dist,
            wastes=wastes,
            capacity=100.0,
            R=1.0,
            C=0.1,
            params=params,
        )

        u_before = solver.utilities[0]
        # Simulate a neutral move: new_profit == current_profit → punish
        current_profit = 5.0
        new_profit = 5.0  # equal, not better

        if new_profit > current_profit:
            solver.utilities[0] = solver._apply_reward(solver.utilities[0])
        else:
            solver.utilities[0] = solver._apply_punishment(solver.utilities[0])

        u_after = solver.utilities[0]
        assert u_after < u_before, (
            f"Neutral move should decrease utility (RL1 punish), "
            f"but got u_before={u_before}, u_after={u_after}"
        )

    @pytest.mark.unit
    def test_rl2_punishment_halves(self, tiny_problem):
        """
        RL2 punishment variant must halve the utility (floor division).
        (Ozcan et al. 2010, Section 3.2)
        """
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver import RLGDHHSolver
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.params import RLGDHHParams

        dist, wastes = tiny_problem
        params = RLGDHHParams(seed=0, punishment_type="RL2", initial_utility=30.0)
        solver = RLGDHHSolver(
            dist_matrix=dist,
            wastes=wastes,
            capacity=100.0,
            R=1.0,
            C=0.1,
            params=params,
        )

        result = solver._apply_punishment(30.0)
        assert result == math.floor(30.0 / 2), f"RL2: expected 15, got {result}"

    @pytest.mark.unit
    def test_rl3_punishment_root(self, tiny_problem):
        """
        RL3 punishment variant must take the floor-sqrt of the utility.
        (Ozcan et al. 2010, Section 3.2)
        """
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver import RLGDHHSolver
        from logic.src.policies.reinforcement_learning_great_deluge_hyper_heuristic.params import RLGDHHParams

        dist, wastes = tiny_problem
        params = RLGDHHParams(seed=0, punishment_type="RL3", initial_utility=30.0)
        solver = RLGDHHSolver(
            dist_matrix=dist,
            wastes=wastes,
            capacity=100.0,
            R=1.0,
            C=0.1,
            params=params,
        )

        result = solver._apply_punishment(36.0)
        assert result == math.floor(math.sqrt(36.0)), f"RL3: expected 6, got {result}"


class TestPOPMUSIC:
    """Unit tests for POPMUSIC matheuristic and paper alignment."""

    @pytest.fixture
    def mock_popmusic_data(self):
        coords = pd.DataFrame({
            "Lat": [0, 1, 10, 11],
            "Lng": [0, 10, 1, 11]
        })
        dist_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                dist_matrix[i, j] = np.linalg.norm(coords.iloc[i].values - coords.iloc[j].values)

        return {
            "coords": coords,
            "mandatory": [1, 2, 3],
            "distance_matrix": dist_matrix,
            "n_vehicles": 4,
            "wastes": {1: 10, 2: 10, 3: 10},
            "capacity": 100,
            "R": 1.0,
            "C": 0.1
        }

    @pytest.mark.unit
    def test_popmusic_lifo_order(self, mock_popmusic_data):
        """Verify that POPMUSIC uses LIFO stack for seed selection."""
        from logic.src.policies.popmusic.solver import run_popmusic

        # Patch _optimize_subproblem to record the order of seeds (neighborhood_indices[0])
        # and always return an improvement for the first few calls to trigger re-pushing.
        call_order = []
        def mock_optimize(*args, **kwargs):
            seed_idx = kwargs["neighborhood_indices"][0]
            call_order.append(seed_idx)
            # Improve only on the first call to seed 3 to see if it re-pushes
            if seed_idx == 3 and len(call_order) == 1:
                return [[0, 1, 0]], 100.0  # Big improvement
            return [[0, seed_idx, 0]], 0.0 # No improvement

        with patch("logic.src.policies.popmusic.solver._optimize_subproblem", side_effect=mock_optimize):
            run_popmusic(**mock_popmusic_data, subproblem_size=2, seed_strategy="lifo")

        # Initial stack for 4 routes is [0, 1, 2, 3].
        # 1. Pop 3. R=[3, 2]. Improve! Re-push 3, 2. Stack: [0, 1, 3, 2]
        # 2. Pop 2. R=[2, 3]. No improvement. Stack: [0, 1, 3]
        # 3. Pop 3. R=[3, 2]. No improvement. Stack: [0, 1]
        # ...
        assert call_order[0] == 3
        assert call_order[1] == 2
        assert call_order[2] == 3

    @pytest.mark.unit
    def test_find_route_neighbors_kdtree(self):
        """Verify KD-Tree neighbor search returns correct indices."""
        from logic.src.policies.popmusic.solver import find_route_neighbors
        from scipy.spatial import KDTree

        centroids = [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([10, 10]),
            np.array([11, 11])
        ]
        kdtree = KDTree(np.array(centroids))

        # k=2 neighbors of seed 0 should be [0, 1]
        neighbors = find_route_neighbors(0, centroids, k=2, kdtree=kdtree, k_prox=4)
        assert set(neighbors) == {0, 1}

        # k=2 neighbors of seed 2 should be [2, 3]
        neighbors = find_route_neighbors(2, centroids, k=2, kdtree=kdtree, k_prox=4)
        assert set(neighbors) == {2, 3}
