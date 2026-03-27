"""
Internal Branch-and-Price-and-Cut (BPC) engine.

Implements exact Branch-and-Price-and-Cut algorithm with:
- Column Generation at every B&B node (LP relaxation)
- Pricing Subproblem (RCSPP via dynamic programming)
- Valid inequalities (Rounded Capacity Cuts for VRPP Set Packing)
- Branch-and-Bound tree with Ryan-Foster branching

References:
    - Barnhart, C., Hane, C. A., & Vance, P. H. (1998).
      "Using Branch-and-Price-and-Cut to Solve Origin-Destination Integer
      Multicommodity Flow Problems." Operations Research, 48(2), 318-326.
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
    - Ryan, D. M., & Foster, B. A. (1981).
      "An Integer Programming Approach to Scheduling".
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..branch_and_cut.separation import SeparationEngine
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.master_problem import Route, VRPPMasterProblem
from ..branch_and_price.rcspp_dp import RCSPPSolver
from ..branch_and_price.ryan_foster_branching import (
    BranchAndBoundTree,
    RyanFosterBranching,
)
from ..other.operators.repair.greedy import greedy_insertion, greedy_profit_insertion


def _separate_rcc(
    master: VRPPMasterProblem,
    sep_engine: SeparationEngine,
    v_model: VRPPModel,
    max_cuts: int,
) -> int:
    """
    Separate and add Rounded Capacity Cuts (RCC) for VRPP Set Packing.

    For VRPP Set Packing, cuts are formulated with exact mathematical relaxation:
        sum_{e in delta(S)} x_e >= 2*k(S) - M * sum_{i in S} (1 - y_i)

    Where:
    - delta(S) is the set of edges crossing the boundary of node set S
    - k(S) = ceil(q(S) / Q) is the minimum number of vehicles needed
    - y_i is the visitation variable (sum of lambda_k covering node i)
    - M is a large constant

    The master problem enforces this by only activating the cut when nodes
    in S have sufficient visitation probability in the LP solution.

    Args:
        master: Master problem instance
        sep_engine: Separation engine for finding violated cuts
        v_model: VRPP model for edge indexing
        max_cuts: Maximum number of cuts to add

    Returns:
        Number of cuts added
    """
    edge_vars = master.get_edge_usage()
    if not edge_vars:
        return 0

    # Note: Node visitation probabilities (y_i) are tracked in the master problem
    # and used by add_set_packing_capacity_cut() to enforce relaxation

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
        if ineq.type == "CAPACITY":
            node_set = list(ineq.node_set)

            # For exact Set Packing, we add the cut with boundary relaxation
            # The cut is: sum_{e in delta(S)} >= 2*k(S)
            # But it only becomes tight when nodes are actually visited
            # This is enforced by the add_set_packing_capacity_cut method
            if master.add_set_packing_capacity_cut(node_set, ineq.rhs):
                added_cuts += 1

    # If we added cuts, re-solve LP and continue
    if added_cuts > 0:
        master.solve_lp_relaxation()
    return added_cuts


def _solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List] = None,
) -> int:
    """
    Solve the pricing subproblem and add positive reduced cost columns.

    Args:
        master: Master problem instance
        pricing_solver: RCSPP solver for pricing
        branching_constraints: Optional Ryan-Foster branching constraints

    Returns:
        Number of columns added
    """
    duals = master.get_reduced_cost_coefficients()
    cut_duals = master.dual_capacity_cuts
    new_columns_data = pricing_solver.solve(
        duals,
        capacity_cut_duals=cut_duals,
        max_routes=5,
        branching_constraints=branching_constraints,
    )

    if not new_columns_data:
        return 0  # No more positive reduced cost columns

    # Add new columns to master
    added = 0
    for r_nodes, red_cost in new_columns_data:
        if red_cost > 1e-4:
            cost, revenue, load, coverage = pricing_solver.compute_route_details(r_nodes)
            route = Route(r_nodes, cost, revenue, load, coverage)

            # Check if route satisfies branching constraints
            if branching_constraints:
                feasible = all(bc.is_route_feasible(route) for bc in branching_constraints)
                if not feasible:
                    continue

            master.add_route(route)
            added += 1
    return added


def _is_solution_integer(route_values: Dict[int, float], tol: float = 1e-6) -> bool:
    """
    Check if LP solution is integer.

    Args:
        route_values: Dictionary of route indices to their LP values
        tol: Numerical tolerance

    Returns:
        True if all values are integer (0 or 1)
    """
    return all(abs(val - round(val)) <= tol for val in route_values.values())


def _column_generation_loop(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    sep_engine: SeparationEngine,
    v_model: VRPPModel,
    branching_constraints: Optional[List],
    max_cg_iterations: int,
    max_cuts: int,
    time_limit: Optional[float],
    start_time: float,
) -> Tuple[float, Dict[int, float]]:
    """
    Run Column Generation + Cutting Plane loop at a B&B node.

    Args:
        master: Master problem
        pricing_solver: RCSPP pricing solver
        sep_engine: Separation engine for cuts
        v_model: VRPP model for separation
        branching_constraints: Active branching constraints
        max_cg_iterations: Maximum CG iterations
        max_cuts: Maximum cuts per iteration
        time_limit: Optional time limit in seconds
        start_time: Process start time

    Returns:
        Tuple of (LP objective value, route values dictionary)
    """
    for _iteration in range(max_cg_iterations):
        if time_limit and (time.process_time() - start_time) > time_limit:
            break

        # Solve LP Relaxation
        try:
            obj_val, route_vals = master.solve_lp_relaxation()
        except Exception as e:
            raise RuntimeError("LP relaxation failed at B&B node") from e

        # A. Separate Cuts (Rounded Capacity Cuts)
        _separate_rcc(master, sep_engine, v_model, max_cuts)

        # B. Solve Pricing Subproblem
        added = _solve_pricing_step(master, pricing_solver, branching_constraints)

        if added == 0:
            # No more columns with positive reduced cost
            break

    # Get final LP solution
    obj_val, route_vals = master.solve_lp_relaxation()
    return obj_val, route_vals


def run_custom_bpc(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    profit_aware_operators: bool = False,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Waste-Collecting CVRP using exact Branch-and-Price-and-Cut.

    This engine implements:
    1. Initial Column Generation via greedy heuristics
    2. Branch-and-Bound tree with Ryan-Foster branching
    3. Column Generation at every B&B node
    4. Exact Pricing via RCSPP
    5. Valid Inequalities: Rounded Capacity Cuts adapted for VRPP Set Packing
    6. Best-first search with bound-based pruning

    Algorithm Overview:
    - Initialize B&B tree with root node
    - While tree is not empty:
        - Pop best node (highest LP bound)
        - Apply branching constraints to master and pricing
        - Run Column Generation + Cutting Plane loop
        - If LP is integer: update incumbent
        - If LP is fractional: branch using Ryan-Foster
        - Prune by bound

    Args:
        dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1)
        wastes: Waste volumes for each node
        capacity: Vehicle capacity
        R: Revenue per kg
        C: Cost per km
        values: Configuration dictionary with keys:
            - max_cg_iterations: Max CG iterations per node (default 50)
            - max_cuts_per_iteration: Max cuts to add per iteration (default 5)
            - max_bb_nodes: Max B&B nodes to explore (default 1000)
            - time_limit: Time limit in seconds (optional)
        mandatory_nodes: List of mandatory node indices
        expand_pool: Whether to expand initial route pool
        profit_aware_operators: Whether to use profit-aware greedy
        recorder: Optional state recorder for tracking

    Returns:
        Tuple of (best_routes, best_cost)
    """
    start_time = time.process_time()
    n_nodes = len(dist_matrix) - 1
    m_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Configuration
    max_cg_iter = values.get("max_cg_iterations", 50)
    max_cuts = values.get("max_cuts_per_iteration", 5)
    max_bb_nodes = values.get("max_bb_nodes", 1000)
    time_limit = values.get("time_limit")

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
    if profit_aware_operators:
        initial_routes_nodes = greedy_profit_insertion(
            [],
            list(range(1, n_nodes + 1)),
            dist_matrix,
            wastes,
            capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            expand_pool=expand_pool,
        )
    else:
        initial_routes_nodes = greedy_insertion(
            [],
            list(range(1, n_nodes + 1)),
            dist_matrix,
            wastes,
            capacity,
            mandatory_nodes=mandatory_nodes,
            expand_pool=expand_pool,
        )

    initial_columns = []
    pricing_helper = RCSPPSolver(n_nodes, dist_matrix, wastes, capacity, R, C, m_set)
    for r_nodes in initial_routes_nodes:
        cost, revenue, load, coverage = pricing_helper.compute_route_details(r_nodes)
        initial_columns.append(Route(r_nodes, cost, revenue, load, coverage))

    master.build_model(initial_columns)

    # 3. Initialize pricing and separation
    pricing_solver = RCSPPSolver(n_nodes, dist_matrix, wastes, capacity, R, C, m_set)
    v_model = VRPPModel(n_nodes + 1, dist_matrix, wastes, capacity, R, C, m_set)
    sep_engine = SeparationEngine(v_model)

    # 4. Initialize Branch-and-Bound Tree
    bb_tree = BranchAndBoundTree()
    nodes_explored = 0

    # 5. Branch-and-Bound Loop
    while not bb_tree.is_empty() and nodes_explored < max_bb_nodes:
        if time_limit and (time.process_time() - start_time) > time_limit:
            break

        # Get next node (best-first)
        current_node = bb_tree.get_next_node()
        if current_node is None:
            break

        nodes_explored += 1

        # Get branching constraints for this node
        branching_constraints = current_node.get_all_constraints()

        # Run Column Generation at this node
        try:
            lp_obj, route_values = _column_generation_loop(
                master=master,
                pricing_solver=pricing_solver,
                sep_engine=sep_engine,
                v_model=v_model,
                branching_constraints=branching_constraints,
                max_cg_iterations=max_cg_iter,
                max_cuts=max_cuts,
                time_limit=time_limit,
                start_time=start_time,
            )
        except RuntimeError:
            # LP infeasible at this node
            current_node.is_infeasible = True
            continue

        # Store LP bound
        current_node.lp_bound = lp_obj
        current_node.route_values = route_values

        # Check if we can prune by bound
        if bb_tree.best_integer_solution is not None and lp_obj <= bb_tree.best_integer_solution:
            # Prune by bound
            continue

        # Check if solution is integer
        if _is_solution_integer(route_values):
            # Integer solution found
            current_node.is_integer = True
            current_node.ip_solution = lp_obj

            # Update incumbent
            if bb_tree.update_incumbent(current_node, lp_obj):
                # Prune dominated nodes
                bb_tree.prune_by_bound()

        else:
            # Solution is fractional - need to branch
            # Use Ryan-Foster branching to find a pair (r, s)
            branching_pair = RyanFosterBranching.find_branching_pair(
                master.routes,
                route_values,
            )

            if branching_pair is None:
                # Couldn't find branching pair (shouldn't happen)
                continue

            node_r, node_s = branching_pair

            # Create two child nodes
            left_child, right_child = RyanFosterBranching.create_child_nodes(
                current_node,
                node_r,
                node_s,
            )

            # Add children to tree
            bb_tree.add_node(left_child)
            bb_tree.add_node(right_child)

    # 6. Extract best integer solution
    if bb_tree.best_integer_node is None:
        # No integer solution found - use initial greedy
        fallback_cost = sum(
            dist_matrix[0][r[0]] + sum(dist_matrix[r[i]][r[i + 1]] for i in range(len(r) - 1)) + dist_matrix[r[-1]][0]
            for r in initial_routes_nodes
        )
        return initial_routes_nodes, fallback_cost

    # Reconstruct solution from best node
    best_node = bb_tree.best_integer_node
    best_routes_objects = []
    for idx, val in best_node.route_values.items():  # type: ignore[union-attr]
        if val > 0.5:  # Binary variable
            best_routes_objects.append(master.routes[idx])

    final_routes = [r.nodes for r in best_routes_objects]
    final_cost = sum(r.cost for r in best_routes_objects)

    if recorder:
        recorder.record(
            engine="exact_bpc",
            iterations=nodes_explored,
            obj_val=bb_tree.best_integer_solution,
            n_routes=len(final_routes),
        )

    return final_routes, final_cost
