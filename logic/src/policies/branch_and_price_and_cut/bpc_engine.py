"""
Internal Branch-and-Price-and-Cut (BPC) engine.
Implements a faithful Column Generation loop with Restricted Master Problem (RMP)
and Exact Pricing Subproblem (RCSPP).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..branch_and_price.master_problem import Route, VRPPMasterProblem
from ..branch_and_price.rcspp_dp import RCSPPSolver


def run_internal_bpc(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Waste-Collecting CVRP using internal Branch-and-Price.

    This engine implements:
    1. Initial Column Generation via greedy heuristics.
    2. LP Relaxation of Master Problem.
    3. Exact Pricing via RCSPP to find positive reduced cost columns.
    4. IP recovery of best column set.
    """
    start_time = time.process_time()
    n_nodes = len(dist_matrix) - 1
    m_set = set(mandatory_nodes) if mandatory_nodes else set()

    # 1. Initialize Master Problem
    master = VRPPMasterProblem(
        n_nodes=n_nodes,
        mandatory_nodes=m_set,
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
    )

    # 2. Initial Columns (Greedy)
    from ..other.operators.repair.greedy import greedy_insertion

    initial_routes_nodes = greedy_insertion(
        [], list(range(1, n_nodes + 1)), dist_matrix, wastes, capacity, R=R, mandatory_nodes=mandatory_nodes
    )

    initial_columns = []
    pricing_helper = RCSPPSolver(n_nodes, dist_matrix, wastes, capacity, R, C, m_set)
    for r_nodes in initial_routes_nodes:
        # Compute details using pricing helper logic
        cost, revenue, load, coverage = pricing_helper.compute_route_details(r_nodes)
        initial_columns.append(Route(r_nodes, cost, revenue, load, coverage))

    master.build_model(initial_columns)

    # 3. Column Generation Loop
    pricing_solver = RCSPPSolver(n_nodes, dist_matrix, wastes, capacity, R, C, m_set)
    max_iter = values.get("max_cg_iterations", 50)
    actual_iters = 0

    for iteration in range(max_iter):
        actual_iters = iteration
        if values.get("time_limit") and (time.process_time() - start_time) > values["time_limit"]:
            break

        # Solve LP Relaxation
        try:
            master.solve_lp_relaxation()
        except Exception:
            break

        # Get dual values and solve pricing
        duals = master.get_reduced_cost_coefficients()
        new_columns_data = pricing_solver.solve(duals, max_routes=5)

        if not new_columns_data:
            break  # No more positive reduced cost columns

        # Add new columns to master
        added = 0
        for r_nodes, red_cost in new_columns_data:
            if red_cost > 1e-4:
                cost, revenue, load, coverage = pricing_solver.compute_route_details(r_nodes)
                master.add_route(Route(r_nodes, cost, revenue, load, coverage))
                added += 1

        if added == 0:
            break

    # 4. Final IP Solution
    try:
        obj_ip, selected_routes_objects = master.solve_ip()
        final_routes = [r.nodes for r in selected_routes_objects]
        final_cost = sum(r.cost for r in selected_routes_objects)
    except Exception:
        # Fallback to greedy if IP fails
        fallback_cost = sum(
            dist_matrix[0][r[0]] + sum(dist_matrix[r[i]][r[i + 1]] for i in range(len(r) - 1)) + dist_matrix[r[-1]][0]
            for r in initial_routes_nodes
        )
        return initial_routes_nodes, fallback_cost

    if recorder:
        recorder.record(engine="internal_bpc", iterations=actual_iters, obj_val=obj_ip, n_routes=len(final_routes))

    return final_routes, final_cost
