import numpy as np
import pytest
from logic.src.policies.hybrid_genetic_search import run_hgs
from logic.src.policies.multi_vehicle import find_routes, find_routes_ortools
from logic.src.policies.policy_vrpp import run_vrpp_optimizer


@pytest.fixture
def base_vrpp_data():
    """Generates a small valid VRPP instance for testing solvers."""
    bins = np.array([50.0, 95.0, 30.0, 85.0, 50.0])  # fill levels
    # 6x6 distance matrix (depot + 5 bins)
    dist_matrix = [
        [0, 10, 10, 10, 10, 10],
        [10, 0, 5, 10, 15, 20],
        [10, 5, 0, 5, 10, 15],
        [10, 10, 5, 0, 5, 10],
        [10, 15, 10, 5, 0, 5],
        [10, 20, 15, 10, 5, 0],
    ]
    values = {
        "Q": 100.0,
        "R": 1.0,
        "B": 1.0,
        "C": 0.1,
        "V": 1.0,
        "Omega": 0.1,
        "delta": 0.0,
        "psi": 0.9,  # Threshold to collect
    }
    binsids = [0, 1, 2, 3, 4, 5]
    must_go = [2, 4]  # Bin IDs (index in dist_matrix)

    return {"bins": bins, "dist_matrix": dist_matrix, "values": values, "binsids": binsids, "must_go": must_go}


class TestVRPPOptimizerContract:
    """Contract tests for run_vrpp_optimizer (Gurobi/Hexaly)."""

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi", "hexaly"])
    def test_vrpp_optimizer_basic_contract(self, base_vrpp_data, backend, check_license):
        data = base_vrpp_data
        routes, profit, cost = run_vrpp_optimizer(
            bins=data["bins"],
            distance_matrix=data["dist_matrix"],
            param=0.0,
            media=np.zeros(len(data["bins"])),
            desviopadrao=np.zeros(len(data["bins"])),
            values=data["values"],
            binsids=data["binsids"],
            must_go=data["must_go"],
            optimizer=backend,
            time_limit=5,
        )

        # 1. Structure Check
        assert isinstance(routes, list), "Routes must be a list"
        assert routes[0] == 0, "Route must start at depot"
        assert routes[-1] == 0, "Route must end at depot"

        # 2. Results Check
        assert isinstance(profit, float), "Profit must be a float"
        assert isinstance(cost, float), "Cost must be a float"

        # 3. Constraint Check (Must-go)
        # must_go contains ID 2 and 4. They MUST be in the routes.
        # Note: ID 0 is depot. Bins are 1-5.
        for mg_id in data["must_go"]:
            assert mg_id in routes, f"Must-go bin {mg_id} was not collected by {backend}"

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi", "hexaly"])
    def test_vrpp_optimizer_empty_bins(self, base_vrpp_data, backend, check_license):
        """Case where no bins should be collected."""
        data = base_vrpp_data
        data["bins"] = np.zeros(len(data["bins"]))  # all empty
        data["must_go"] = []
        data["values"]["psi"] = 0.99

        routes, profit, cost = run_vrpp_optimizer(
            bins=data["bins"],
            distance_matrix=data["dist_matrix"],
            param=0.0,
            media=np.zeros(len(data["bins"])),
            desviopadrao=np.zeros(len(data["bins"])),
            values=data["values"],
            binsids=data["binsids"],
            must_go=data["must_go"],
            optimizer=backend,
            time_limit=5,
        )
        # Should just return to depot [0] or [0, 0]
        assert set(routes) == {0}


class TestMultiVehicleContract:
    """Contract tests for find_routes (PyVRP/OR-Tools)."""

    @pytest.mark.integration
    def test_pyvrp_contract(self, base_vrpp_data):
        data = base_vrpp_data
        # find_routes expects demands for clients (1..N)
        # our bins array has 5 elements for bins 1..5
        demands = data["bins"]  # already 5 elements
        to_collect = [1, 2, 3, 4, 5]

        tour = find_routes(
            dist_mat=np.array(data["dist_matrix"]),
            demands=demands,
            max_caps=100,
            to_collect=to_collect,
            n_vehicles=0,  # Unlimited
        )

        assert isinstance(tour, list)
        assert tour[0] == 0
        assert tour[-1] == 0
        assert all(id in tour for id in to_collect), "All bins should be visited"

    @pytest.mark.integration
    def test_ortools_contract(self, base_vrpp_data):
        data = base_vrpp_data
        demands = data["bins"]
        to_collect = [1, 2, 3, 4, 5]

        tour = find_routes_ortools(
            dist_mat=np.array(data["dist_matrix"]),
            demands=demands,
            max_caps=100,
            to_collect=to_collect,
            n_vehicles=0,  # Unlimited
        )

        assert isinstance(tour, list)
        assert tour.count(0) >= 3, "Should have at least Start-End and one separator for 2 vehicles or splits"
        assert all(id in tour for id in to_collect)


class TestHGSSolverContract:
    """Contract tests for run_hgs."""

    @pytest.mark.integration
    @pytest.mark.parametrize("engine", ["custom", "pyvrp"])
    def test_hgs_contract(self, base_vrpp_data, engine):
        data = base_vrpp_data
        # hgs expects demands as dict
        demands_dict = {i: data["bins"][i - 1] for i in range(1, 6)}

        # run_hgs(dist_matrix, demands, capacity, R, C, values, *args)
        values = data["values"]
        values["engine"] = engine
        values["time_limit"] = 2

        routes, profit, cost = run_hgs(
            np.array(data["dist_matrix"]), demands_dict, values["Q"], values["R"], values["C"], values
        )

        assert isinstance(routes, list)
        if routes and isinstance(routes[0], list):
            # nested routes [[...], [...]]
            flat = [0]
            for r in routes:
                flat.extend(r)
                flat.append(0)
            assert all(id in flat for id in [1, 2, 3, 4, 5])
        else:
            # flat route
            assert all(id in routes for id in [1, 2, 3, 4, 5])
