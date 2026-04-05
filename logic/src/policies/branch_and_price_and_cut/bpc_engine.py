"""
Internal Branch-and-Price-and-Cut (BPC) engine.

This module provides an exact Branch-and-Price-and-Cut (BPC) solver adapted for the
Vehicle Routing Problem with Profits (VRPP). While it borrows the high-level
algorithmic sequencing from the BPC framework for multicommodity flow (Barnhart et al., 1998),
it is specifically tailored for the VRPP context.

Key Features:
- Adapts BPC sequencing: Column generation to convergence, followed by cut separation,
  followed by branching.
- Pricing Subproblem: Solves the Resource-Constrained Shortest Path Problem (RCSPP)
  via dynamic programming.
- Branching: Employs Ryan-Foster branching, which is suited for VRPP as it
  directly modifies the pricing problem's state space.
- Valid Inequalities: Includes Rounded Capacity Cuts (RCC) for the VRPP context.

References:
    - Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W., & Vance, P. H. (1998).
      "Branch-and-price: Column generation for solving huge integer programs."
      Operations Research, 46(3), 316-329.
    - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004).
      "A new branch-and-cut algorithm for the capacitated vehicle routing problem."
    - Ryan, D. M., & Foster, B. A. (1981).
      "An Integer Programming Approach to Scheduling".
"""

import logging
import time
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..branch_and_cut.separation import SeparationEngine
from ..branch_and_cut.vrpp_model import VRPPModel
from ..branch_and_price.branching import BranchAndBoundTree
from ..branch_and_price.master_problem import Route, VRPPMasterProblem
from ..branch_and_price.rcspp_dp import RCSPPSolver
from ..other.operators.repair.greedy import greedy_insertion, greedy_profit_insertion
from .cutting_planes import CuttingPlaneEngine, create_cutting_plane_engine
from .search_strategy import create_search_strategy

logger = logging.getLogger(__name__)


def _apply_branching_to_master(
    master: VRPPMasterProblem,
    branching_constraints: List,
) -> None:
    """
    Filter the Master Problem column pool by disabling routes that violate
    branching constraints at the current B&B node.

    Column Contamination Problem:
    ------------------------------
    The Master Problem maintains a global column pool across the entire B&B tree.
    When we branch at a node, we add constraints (e.g., "nodes r and s must be
    together" or "arc (u,v) is forbidden"). However, the existing column pool
    may contain routes generated at ancestor nodes that violate these new constraints.

    If we don't filter these routes, the LP solver will select them, producing
    fractional solutions that violate the branching decisions. This causes:
    1. Incorrect LP bounds (too optimistic)
    2. Infinite branching loops (same fractional solution reappears)
    3. Invalid integer solutions

    Solution:
    ---------
    We temporarily disable violating routes by setting their upper bound to 0:
        var.UB = 0.0

    This is more efficient than:
    - Deleting and re-adding columns (expensive Gurobi operations)
    - Maintaining separate column pools per node (memory intensive)
    - Adding explicit branching constraints to Gurobi (creates dense constraint matrix)

    The routes remain in the model structure but cannot be selected in the LP solution.

    Implementation:
    ---------------
    For each route k and its corresponding Gurobi variable λ_k:
    1. Reset upper bound to 1.0 (clean state for this node)
    2. Check if route satisfies all active branching constraints
    3. If any constraint is violated, set var.UB = 0.0 to disable the route

    Args:
        master: Master problem instance with Gurobi model
        branching_constraints: Complete list of branching constraints from root
            to the current node (all ancestors + current node). Must NOT be
            just the current node's constraint — ancestor constraints are needed
            to correctly filter the global column pool.

    References:
    -----------
    Ryan-Foster branching (1981) is used here because it appropriately modifies the
    Resource-Constrained Shortest Path Problem (RCSPP) used for VRPP pricing by
    forbidding or enforcing pairs of nodes in the generated routes.
    This diverges from Barnhart et al. (1998), who utilized divergence branching
    to preserve simple shortest paths in the context of Origin-Destination
    Integer Multicommodity Flow (ODIMCF).
    """
    if not branching_constraints or master.model is None:
        return

    for route, var in zip(master.routes, master.lambda_vars):
        # Reset to clean state (allows reactivation when moving to different branches)
        var.UB = 1.0

        # Check feasibility against all active branching constraints
        is_feasible = all(bc.is_route_feasible(route) for bc in branching_constraints)

        if not is_feasible:
            # Temporarily disable this route by forcing its upper bound to 0
            var.UB = 0.0

    # Update the model to reflect the bound changes
    master.model.update()


def _separate_cuts(
    master: VRPPMasterProblem,
    cut_engine: CuttingPlaneEngine,
    max_cuts: int,
) -> int:
    """
    Separate and add valid inequalities using the configured cutting plane engine.

    This is a modular wrapper that delegates to the specific cutting plane
    engine (RCC, LCI, etc.) configured by the user.

    Args:
        master: Master problem instance
        cut_engine: Cutting plane separation engine
        max_cuts: Maximum number of cuts to add

    Returns:
        Number of cuts added
    """
    return cut_engine.separate_and_add_cuts(master, max_cuts)


def _solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List] = None,
    max_routes: int = 5,
    optimality_gap: float = 1e-4,
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

    # Merge capacity cut duals and SEC duals into a single cut-dual mapping.
    # Both cut families impose crossing penalties on routes that violate them.
    # If the same node set exists in both, their duals are summed.
    all_cut_duals: Dict[FrozenSet[int], float] = {}
    for node_set, dual in master.dual_capacity_cuts.items():
        all_cut_duals[node_set] = all_cut_duals.get(node_set, 0.0) + dual
    for node_set, dual in master.dual_sec_cuts.items():
        all_cut_duals[node_set] = all_cut_duals.get(node_set, 0.0) + dual

    new_columns_data = pricing_solver.solve(
        duals,
        max_routes=max_routes,
        branching_constraints=branching_constraints,
        capacity_cut_duals=all_cut_duals,
    )

    if not new_columns_data:
        return 0  # No more positive reduced cost columns

    # Add new columns to master
    added = 0
    for r_nodes, red_cost in new_columns_data:
        if red_cost > optimality_gap:
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
    cut_engine: CuttingPlaneEngine,
    branching_constraints: Optional[List],
    max_cg_iterations: int,
    max_cuts: int,
    time_limit: Optional[float],
    start_time: float,
    max_routes_per_pricing: int = 5,
    vehicle_limit: Optional[int] = None,
    optimality_gap: float = 1e-4,
    early_termination_gap: float = 1e-3,
) -> Tuple[float, Dict[int, float]]:
    """
    Run Column Generation + Cutting Plane loop at a B&B node.

    CORRECTED SEQUENCING (Barnhart et al. 1998, Section 3):
    --------------------------------------------------------
    1. Price out columns until LP is optimal (no more positive reduced cost)
    2. Attempt to find violated cuts in the current LP solution
    3. If cuts are found, add them and return to Step 1
    4. Terminate when no columns price out AND no cuts are violated

    COLUMN GENERATION EARLY TERMINATION:
    -------------------------------------
    We terminate Phase 1 CG early if the maximum reduced cost of any newly
    generated column, scaled by the vehicle limit, is below the gap tolerance:
        max_rc * K < early_termination_gap
    This is a practical convergence heuristic — not a strict Lagrangian bound —
    that avoids unnecessary LP solves when the remaining improvement is negligible.
    Note: A true Lagrangian upper bound requires the most positive reduced cost
    across the ENTIRE unprice column space, which is not cheaply available.
    """
    needs_resolve = False
    obj_val: float = 0.0
    route_vals: Dict[int, float] = {}
    converged = False
    _iteration = 0

    if max_cg_iterations <= 0:
        raise ValueError(f"max_cg_iterations must be >= 1, got {max_cg_iterations}")

    for _iteration in range(max_cg_iterations):
        if time_limit and (time.process_time() - start_time) > time_limit:
            needs_resolve = True
            break

        # PHASE 1: Column Generation (price until convergence)
        while True:
            try:
                obj_val, route_vals = master.solve_lp_relaxation()
            except Exception as e:
                raise RuntimeError("LP relaxation failed at B&B node") from e

            added = _solve_pricing_step(
                master,
                pricing_solver,
                branching_constraints,
                max_routes=max_routes_per_pricing,
                optimality_gap=optimality_gap,
            )

            if added == 0:
                break

            if hasattr(pricing_solver, "last_max_rc"):
                max_rc = pricing_solver.last_max_rc
                limit = vehicle_limit if vehicle_limit is not None else master.n_nodes
                if max_rc * limit < early_termination_gap:
                    logger.info(
                        f"CG early termination: max_rc * limit = {max_rc * limit:.6f} < {early_termination_gap}"
                    )
                    break

        # PHASE 2: Cutting Planes (separate on converged LP solution)
        cuts_added = _separate_cuts(master, cut_engine, max_cuts)

        if cuts_added == 0:
            converged = True
            break

        needs_resolve = True

    # Warn only when the iteration cap truncated an unconverged loop
    if not converged and _iteration == max_cg_iterations - 1:
        import warnings

        warnings.warn(
            f"CG+Cut loop hit max_cg_iterations={max_cg_iterations} without full convergence. "
            "LP bound at this B&B node may be weaker than optimal. Consider increasing "
            "max_cg_iterations for tighter bounds.",
            stacklevel=3,
        )

    if needs_resolve:
        obj_val, route_vals = master.solve_lp_relaxation()

    return obj_val, route_vals


def run_custom_bpc(  # noqa: C901
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

    This engine implements the Barnhart et al. (1998) BPC framework with
    configurable algorithmic strategies:

    1. Initial Column Generation via greedy heuristics
    2. Branch-and-Bound tree with pluggable branching strategies
    3. Column Generation at every B&B node (following exact BPC sequencing)
    4. Exact Pricing via RCSPP
    5. Modular Valid Inequalities (RCC)
    6. Configurable search strategy (Depth-First search is preferred for basis reuse)

    Algorithm Overview:
    - Initialize B&B tree with root node and selected strategy
    - While tree is not empty:
        - Select next node using configured search strategy
        - Apply branching constraints to master and pricing
        - Run Column Generation + Cutting Plane loop (corrected sequencing)
        - If LP is integer: update incumbent
        - If LP is fractional: branch using configured branching rule
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
            - search_strategy: "best_first" or "depth_first" (default "depth_first")
            - cutting_planes: "rcc" (default "rcc")
            - branching_strategy: "ryan_foster", "edge", or "divergence" (default "divergence")
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
    max_routes_per_pricing = values.get("max_routes_per_pricing", 5)
    max_bb_nodes = values.get("max_bb_nodes", 1000)
    time_limit = values.get("time_limit")

    # Strategy Configuration
    search_strategy_name = values.get("search_strategy", "depth_first")
    cutting_planes_name = values.get("cutting_planes", "rcc")
    branching_strategy_name = values.get("branching_strategy", "divergence")

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
    n_total_nodes = n_nodes + 1  # VRPPModel counts total nodes including depot (index 0)
    v_model = VRPPModel(
        n_nodes=n_total_nodes,
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
        mandatory_nodes=m_set,
    )
    sep_engine = SeparationEngine(v_model)

    # 4. Initialize strategy modules
    search_strategy = create_search_strategy(search_strategy_name)
    cut_engine = create_cutting_plane_engine(cutting_planes_name, v_model, sep_engine)

    # 5. Initialize Branch-and-Bound Tree with configured branching strategy
    bb_tree = BranchAndBoundTree(strategy=branching_strategy_name)
    # Node selection is handled exclusively by the external search_strategy object.
    # bb_tree.get_next_node() is NOT used; always call search_strategy.select_node().
    nodes_explored = 0

    # 6. Branch-and-Bound Loop
    while not bb_tree.is_empty() and nodes_explored < max_bb_nodes:
        if time_limit and (time.process_time() - start_time) > time_limit:
            break

        # Get next node using configured search strategy
        current_node = search_strategy.select_node(bb_tree.open_nodes)
        if current_node is None:
            break

        nodes_explored += 1

        # Get ALL branching constraints along the path from root to this node
        # (ancestors + this node). _apply_branching_to_master resets all UBs
        # to 1.0 and re-filters against the full constraint set, so it requires
        # the complete ancestor chain — not just this node's local constraint.
        branching_constraints = current_node.get_all_constraints()

        # Apply branching constraints to master problem (filter invalid columns)
        _apply_branching_to_master(master, branching_constraints)

        # Run Column Generation at this node with corrected sequencing
        try:
            lp_obj, route_values = _column_generation_loop(
                master=master,
                pricing_solver=pricing_solver,
                cut_engine=cut_engine,
                branching_constraints=branching_constraints,
                max_cg_iterations=max_cg_iter,
                max_cuts=max_cuts,
                time_limit=time_limit,
                start_time=start_time,
                max_routes_per_pricing=max_routes_per_pricing,
                vehicle_limit=master.vehicle_limit,
                optimality_gap=values.get("optimality_gap", 1e-4),
                early_termination_gap=values.get("early_termination_gap", 1e-3),
            )
        except RuntimeError:
            # LP infeasible at this node
            current_node.is_infeasible = True
            continue

        # Store LP bound
        current_node.lp_bound = lp_obj
        current_node.route_values = route_values

        # Check if solution is integer FIRST — before pruning
        if _is_solution_integer(route_values):
            current_node.is_integer = True
            current_node.ip_solution = lp_obj

            # Update incumbent (records equal or better solutions)
            if bb_tree.update_incumbent(current_node, lp_obj):
                bb_tree.prune_by_bound()

            # After recording, prune this node (no need to branch)
            continue

        # Check if we can prune by bound (strictly less than incumbent, since integer
        # solutions equal to incumbent were already handled above)
        if bb_tree.best_integer_solution is not None and lp_obj < bb_tree.best_integer_solution:
            continue

        # Solution is fractional - branch
        children = bb_tree.branch(current_node, master.routes, route_values)
        if children is None:
            # Primary branching strategy found no candidate (e.g., divergence branching
            # requires >=2 outgoing arcs at a divergence node but none exist).
            # Fall back to edge branching to avoid silently abandoning this fractional node.
            logger.warning(
                f"Primary branching strategy '{branching_strategy_name}' returned no "
                "branching candidate at a fractional node. Falling back to edge branching."
            )
            from ..branch_and_price.branching import EdgeBranching

            arc = EdgeBranching.find_branching_arc(master.routes, route_values)
            if arc is None:
                # Truly no branching candidate — node is degenerate, skip it.
                logger.warning("Edge branching fallback also found no candidate. Skipping node.")
                continue
            u, v = arc
            left_child, right_child = EdgeBranching.create_child_nodes(current_node, u, v)
            children = (left_child, right_child)

        left_child, right_child = children
        bb_tree.add_node(left_child)
        bb_tree.add_node(right_child)

    # 6. Extract best integer solution
    if bb_tree.best_integer_node is None:
        # No integer solution found - use initial greedy
        fallback_routes_profit = 0.0
        for r_nodes in initial_routes_nodes:
            cost, revenue, load, coverage = pricing_helper.compute_route_details(r_nodes)
            fallback_routes_profit += revenue - cost
        return initial_routes_nodes, fallback_routes_profit

    # Reconstruct solution from best node
    best_node = bb_tree.best_integer_node
    best_routes_objects = []
    for idx, val in best_node.route_values.items():  # type: ignore[union-attr]
        if val > 0.5:  # Binary variable
            best_routes_objects.append(master.routes[idx])

    final_routes = [r.nodes for r in best_routes_objects]
    final_cost = sum(r.profit for r in best_routes_objects)

    if recorder:
        recorder.record(
            engine="exact_bpc",
            iterations=nodes_explored,
            obj_val=bb_tree.best_integer_solution,
            n_routes=len(final_routes),
        )

    return final_routes, final_cost
