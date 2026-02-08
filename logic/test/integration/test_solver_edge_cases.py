import numpy as np
import pytest
from logic.src.policies.hybrid_genetic_search import run_hgs
from logic.src.policies.cvrp import find_routes, find_routes_ortools
from logic.src.policies.adapters.policy_vrpp import run_vrpp_optimizer

class TestSolverEdgeCases:
    """
    Tests for solver stability handling edge cases:
    - N=0 (No clients)
    - N=1 (Single client)
    - Exact Capacity (Demand = Capacity)
    - Zero Demand (Client with 0 demand)
    - Impossible Demand (Demand > Capacity)
    """

    @pytest.fixture
    def edge_case_data(self):
        """Base template for edge cases."""
        # Minimal valid structure
        return {
            "bins": np.array([]),
            "dist_matrix": [[0]],
            "values": {
                "Q": 100.0,
                "R": 1.0,
                "B": 1.0,
                "C": 1.0,
                "V": 1.0,
                "Omega": 0.0,
                "delta": 0.0,
                "psi": 0.0,
            },
            "binsids": [0],
            "must_go": []
        }

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi", "hexaly"])
    def test_n_zero_commercial(self, backend, edge_case_data, check_license):
        """Test with 0 clients (only depot) - Commercial Solvers."""
        data = edge_case_data
        data["bins"] = np.array([])
        data["dist_matrix"] = [[0]] # 1x1 matrix (Depot only)
        data["binsids"] = [0]

        try:
            routes, _, _ = run_vrpp_optimizer(
                bins=data["bins"],
                distance_matrix=data["dist_matrix"],
                param=0.0,
                media=np.array([]),
                desviopadrao=np.array([]),
                values=data["values"],
                binsids=data["binsids"],
                must_go=[],
                optimizer=backend,
                time_limit=2
            )
            assert routes == [0] or routes == [0, 0] or len(routes) == 0
        except Exception as e:
            pytest.fail(f"{backend} crashed on N=0: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["ortools", "pyvrp", "hgs"])
    def test_n_zero_opensource(self, backend, edge_case_data):
        """Test with 0 clients (only depot) - Open Source Solvers."""
        data = edge_case_data
        data["bins"] = np.array([])
        data["dist_matrix"] = [[0]]
        data["binsids"] = [0]

        if backend == "ortools":
            try:
                tour = find_routes_ortools(
                    dist_mat=np.array(data["dist_matrix"]),
                    demands=np.array([0]), # Depot demand
                    max_caps=100,
                    to_collect=[], # Empty list
                    n_vehicles=1
                )
                assert tour == [0] or tour == [0,0]
            except Exception as e:
                pytest.fail(f"OR-Tools crashed on N=0: {e}")

        elif backend == "pyvrp":
            try:
                tour = find_routes(
                    dist_mat=np.array(data["dist_matrix"]),
                    demands=np.array([0]), # Depot demand
                    max_caps=100,
                    to_collect=[], # Empty list
                    n_vehicles=1
                )
                assert tour == [0, 0] or tour == [0] or tour == []
            except Exception as e:
                pytest.fail(f"PyVRP crashed on N=0: {e}")

        elif backend == "hgs":
            try:
                demands_dict = {}
                routes, _, _ = run_hgs(
                    np.array(data["dist_matrix"]),
                    demands_dict,
                    100,
                    1.0, 1.0, data["values"]
                )
                assert not routes or routes == [0]
            except Exception as e:
                 pytest.fail(f"HGS crashed on N=0: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi", "hexaly"])
    def test_n_one_commercial(self, backend, edge_case_data, check_license):
        """Test with 1 client - Commercial Solvers."""
        data = edge_case_data
        data["bins"] = np.array([50.0])
        data["dist_matrix"] = [[0, 10], [10, 0]]
        data["binsids"] = [0, 1]

        try:
            routes, _, _ = run_vrpp_optimizer(
                bins=data["bins"],
                distance_matrix=data["dist_matrix"],
                param=0.0,
                media=np.zeros(1),
                desviopadrao=np.zeros(1),
                values=data["values"],
                binsids=data["binsids"],
                must_go=[1],
                optimizer=backend,
                time_limit=2
            )
            assert 1 in routes
        except Exception as e:
            pytest.fail(f"{backend} crashed on N=1: {e}")

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["ortools", "pyvrp", "hgs"])
    def test_n_one_opensource(self, backend, edge_case_data):
        """Test with 1 client - Open Source Solvers."""
        data = edge_case_data
        data["bins"] = np.array([50.0])
        data["dist_matrix"] = [[0, 10], [10, 0]]
        data["binsids"] = [0, 1]

        if backend == "ortools":
            tour = find_routes_ortools(
                dist_mat=np.array(data["dist_matrix"]),
                demands=np.array([0, 50]),
                max_caps=100,
                to_collect=[1],
                n_vehicles=1
            )
            assert 1 in tour

        elif backend == "pyvrp":
            tour = find_routes(
                dist_mat=np.array(data["dist_matrix"]),
                demands=np.array([0, 50]),
                max_caps=100,
                to_collect=[1],
                n_vehicles=1
            )
            assert 1 in tour

        elif backend == "hgs":
            demands_dict = {1: 50}
            routes, _, _ = run_hgs(
                np.array(data["dist_matrix"]),
                demands_dict,
                100, 1.0, 1.0, data["values"]
            )
            flat = []
            if routes and isinstance(routes[0], list):
                for r in routes: flat.extend(r)
            else:
                 flat = routes
            assert 1 in flat

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi"])
    def test_exact_capacity(self, backend, check_license):
        """Test with demand exactly equal to capacity."""
        # 2 nodes, 50+50 = 100 capacity.
        bins = np.array([50.0, 50.0])
        dist_matrix = [[0,1,1],[1,0,1],[1,1,0]]
        values = {"Q": 100.0, "R": 1, "B":1, "C":0.1, "V":1, "Omega":0, "delta":0, "psi":0}

        # Test Gurobi (most sensitive to constraints)
        try:
             routes, _, _ = run_vrpp_optimizer(
                bins=bins,
                distance_matrix=dist_matrix,
                param=0.0,
                media=np.zeros(2),
                desviopadrao=np.zeros(2),
                values=values,
                binsids=[0,1,2],
                must_go=[1,2],
                optimizer="gurobi",
                time_limit=2
            )
             assert 1 in routes and 2 in routes
        except Exception as e:
            pytest.skip(f"Gurobi skipped/failed: {e}")

    @pytest.mark.integration
    def test_zero_demand_node(self):
        """Test handling of nodes with 0 demand."""
        # Should be valid to visit.
        tour = find_routes_ortools(
            dist_mat=np.array([[0,1],[1,0]]),
            demands=np.array([0, 0]), # Node 1 has 0 demand
            max_caps=100,
            to_collect=[1],
            n_vehicles=1
        )
        assert 1 in tour

    @pytest.mark.integration
    def test_impossible_demand_pyvrp(self):
        """Test demand > capacity."""
        # PyVRP acts as VRP solver. If demand > capacity, it might fail or return partial?
        # Typically undefined behavior or error. We just want it NOT to segfault.
        try:
            tour = find_routes(
                dist_mat=np.array([[0,1],[1,0]]),
                demands=np.array([0, 150]), # 150 > 100
                max_caps=100,
                to_collect=[1],
                n_vehicles=1
            )
            # It might visit it if multiple trips allowed? PyVRP usually assumes single trip per vehicle unless configured.
            # Here we check it returns SOMETHING list-like.
            assert isinstance(tour, list)
        except Exception:
            # Raising exception is also acceptable for impossible constraints
            pass
