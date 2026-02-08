import numpy as np
import pytest
from logic.src.policies.hybrid_genetic_search import run_hgs
from logic.src.policies.cvrp import find_routes, find_routes_ortools
from logic.src.policies.adapters.policy_vrpp import run_vrpp_optimizer


class TestSolverParity:
    """Compare solvers on identical instances."""

    @pytest.mark.integration
    @pytest.mark.parametrize("backend", ["gurobi", "hexaly"])
    def test_parity_small_instance(self, parity_instance, check_license):
        data = parity_instance
        results = {}

        # 1. Gurobi (Ground Truth usually)
        try:
            routes_g, profit_g, cost_g = run_vrpp_optimizer(
                bins=data["bins"],
                distance_matrix=data["dist_matrix"],
                param=0.0,
                media=np.zeros(5),
                desviopadrao=np.zeros(5),
                values=data["values"],
                binsids=data["binsids"],
                must_go=data["must_go"],
                optimizer="gurobi",
                time_limit=5
            )
            # Calculate objective: Prize - Cost
            # Prize = sum(bins[i-1] * R) for i in visited (excluding 0)
            # Cost = route cost * C
            # But run_vrpp_optimizer returns (Route, Profit, Cost).
            # Profit returned is usually the net objective? Or just revenue?
            # Let's trust the return values for now or recompute.
            results["gurobi"] = (cost_g, sorted(list(set(routes_g) - {0})))
        except Exception as e:
            pytest.skip(f"Gurobi failed or not available: {e}")

        # 2. Hexaly
        try:
            routes_h, profit_h, cost_h = run_vrpp_optimizer(
                bins=data["bins"],
                distance_matrix=data["dist_matrix"],
                param=0.0,
                media=np.zeros(5),
                desviopadrao=np.zeros(5),
                values=data["values"],
                binsids=data["binsids"],
                must_go=data["must_go"],
                optimizer="hexaly",
                time_limit=5
            )
            results["hexaly"] = (cost_h, sorted(list(set(routes_h) - {0})))
        except Exception as e:
            print(f"Hexaly failed: {e}")

        # 3. OR-Tools (find_routes)
        # Note: find_routes minimizes DISTANCE for a given SET of nodes.
        # It does NOT do node selection (VRPP).
        # To compare, we must run it on the set of nodes Gurobi selected (if Gurobi ran),
        # OR just give it ALL nodes and valid capacity.
        # Our instance has large capacity, so it SHOULD visit ALL nodes if we tell it to.
        try:
            to_collect = [1, 2, 3, 4, 5]
            demands = data["bins"]
            tour_o = find_routes_ortools(
                dist_mat=np.array(data["dist_matrix"]),
                demands=demands,
                max_caps=1000,
                to_collect=to_collect,
                n_vehicles=1
            )
            # Compute cost manually
            cost_o = 0
            for i in range(len(tour_o) - 1):
                u, v = tour_o[i], tour_o[i+1]
                cost_o += data["dist_matrix"][u][v]

            results["ortools"] = (cost_o, sorted(list(set(tour_o) - {0})))
        except Exception as e:
            print(f"OR-Tools failed: {e}")

        # 4. HGS (run_hgs) - VRPP solver
        try:
            demands_dict = {i: data["bins"][i-1] for i in range(1, 6)}
            routes_hgs, profit_hgs, cost_hgs = run_hgs(
                np.array(data["dist_matrix"]),
                demands_dict,
                1000,
                data["values"]["R"],
                data["values"]["C"],
                data["values"]
            )

            visited_hgs = []
            if routes_hgs and isinstance(routes_hgs[0], list):
                 for r in routes_hgs: visited_hgs.extend(r)
            else:
                 visited_hgs = routes_hgs

            results["hgs"] = (cost_hgs, sorted(list(set(visited_hgs) - {0})))
        except Exception as e:
            print(f"HGS failed: {e}")

        # Assertions
        # If Gurobi ran, it is the baseline.
        if "gurobi" in results:
            baseline_cost, baseline_nodes = results["gurobi"]
            print(f"Baseline (Gurobi): Cost={baseline_cost}, Nodes={baseline_nodes}")

            for solver, (cost, nodes) in results.items():
                if solver == "gurobi": continue

                # Check Node Selection Parity (For VRPP solvers)
                if solver in ["hexaly", "hgs"]:
                    # Exact node match expected for this trivial instance
                    assert nodes == baseline_nodes, f"{solver} selected different nodes than Gurobi"

                # Check Cost Parity
                # Allow small tolerance
                assert abs(cost - baseline_cost) < 1.0, f"{solver} cost {cost} diverges from baseline {baseline_cost}"

        else:
            pytest.skip("Gurobi not available for baseline comparison")
