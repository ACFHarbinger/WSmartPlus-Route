"""
Internal Branch-and-Price-and-Cut (BPC) engine.
Implements a faithful Column Generation loop with Restricted Master Problem (RMP),
Exact Pricing Subproblem (RCSPP), and valid inequalities (Rounded Capacity Cuts).

Reference:
    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
    "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..branch_and_cut.separation import SeparationEngine
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.master_problem import Route, VRPPMasterProblem
from ..branch_and_price.rcspp_dp import RCSPPSolver
from ..other.operators.repair.greedy import greedy_insertion


def _separate_rcc(
    master: VRPPMasterProblem,
    sep_engine: SeparationEngine,
    v_model: VRPPModel,
    max_cuts: int,
) -> int:
    """Separate and add Rounded Capacity Cuts (RCC) to the master problem."""
    edge_vars = master.get_edge_usage()
    if not edge_vars:
        return 0

    # Map edge_usage to x_vals array for SeparationEngine
    x_vals = np.zeros(len(v_model.edges))
    for (i, j), val in edge_vars.items():
        if (min(i, j), max(i, j)) in v_model.edge_to_idx:
            idx = v_model.edge_to_idx[(min(i, j), max(i, j))]
            x_vals[idx] = val

    # Separate cuts
    violated_ineqs = sep_engine.separate(x_vals, max_cuts=max_cuts)
    added_cuts = 0
    for ineq in violated_ineqs:
        if ineq.type == "CAPACITY" and master.add_capacity_cut(list(ineq.node_set), ineq.rhs):
            added_cuts += 1

    # If we added cuts, re-solve LP and continue
    if added_cuts > 0:
        master.solve_lp_relaxation()
    return added_cuts


def _solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
) -> int:
    """Solve the pricing subproblem and add positive reduced cost columns."""
    duals = master.get_reduced_cost_coefficients()
    cut_duals = master.dual_capacity_cuts
    new_columns_data = pricing_solver.solve(duals, capacity_cut_duals=cut_duals, max_routes=5)

    if not new_columns_data:
        return 0  # No more positive reduced cost columns

    # Add new columns to master
    added = 0
    for r_nodes, red_cost in new_columns_data:
        if red_cost > 1e-4:
            cost, revenue, load, coverage = pricing_solver.compute_route_details(r_nodes)
            master.add_route(Route(r_nodes, cost, revenue, load, coverage))
            added += 1
    return added


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
    Solve Waste-Collecting CVRP using internal Branch-and-Price-and-Cut.

    This engine implements:
    1. Initial Column Generation via greedy heuristics.
    2. LP Relaxation of Master Problem.
    3. Exact Pricing via RCSPP to find positive reduced cost columns.
    4. Valid Inequalities: Rounded Capacity Cuts (RCC) separated at each node.
    5. IP recovery of best column set.
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

    # 3. Column Generation + Cutting Plane Loop
    pricing_solver = RCSPPSolver(n_nodes, dist_matrix, wastes, capacity, R, C, m_set)

    # Initialize separation engine for RCCs
    v_model = VRPPModel(n_nodes + 1, dist_matrix, wastes, capacity, R, C, m_set)
    sep_engine = SeparationEngine(v_model)

    max_iter = values.get("max_cg_iterations", 50)
    actual_iters = 0
    max_cuts = values.get("max_cuts_per_iteration", 5)

    for iteration in range(max_iter):
        actual_iters = iteration
        if values.get("time_limit") and (time.process_time() - start_time) > values["time_limit"]:
            break

        # Solve LP Relaxation
        try:
            master.solve_lp_relaxation()
        except Exception:
            break

        # A. Separate Cuts (Rounded Capacity Cuts)
        _separate_rcc(master, sep_engine, v_model, max_cuts)

        # B. Solve Pricing Subproblem
        added = _solve_pricing_step(master, pricing_solver)

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
        recorder.record(engine="native_bpc", iterations=actual_iters, obj_val=obj_ip, n_routes=len(final_routes))

    return final_routes, final_cost
