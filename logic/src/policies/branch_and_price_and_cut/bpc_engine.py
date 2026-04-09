r"""
Branch-and-Price-and-Cut (BPC) Engine for VRPP.

This engine implements a state-of-the-art exact solver for the Vehicle Routing
Problem with Profits (VRPP), synthesizing the Origin-Destination Integer
Multicommodity Flow (ODIMCF) sequencing framework of Barnhart et al. (1998)
with advanced polyhedral cutting planes (RCC, SRI, LCI).

Algorithmic Components:
-----------------------
- Master Problem: Set Partitioning formulation using Gurobi with Column Generation.
- Pricing Subproblem: Elementary / ng-relaxed Resource Constrained Shortest Path
  Problem (RCSPP) solved via DP.
- Branching: Multi-Edge Spatial Partitioning and exact Ryan-Foster co-occurrence.
- Cutting Planes: Dynamically separated Rounded Capacity Cuts (RCC), Subset-Row
  Inequalities (SRI), and Lifted Cover Inequalities (LCI).

References:
-----------
- Barnhart, C., Johnson, E. L., Nemhauser, G. L., Savelsbergh, M. W., & Sigismondi, P. H.
  (1998). "Branch-and-price: Column generation for solving huge integer programs."
  Operations Research, 46(3), 316-329.
- Barnhart, C., Hane, C. A., & Vance, P. H. (2000). "Using branch-and-price-and-cut
  to solve origin-destination integer multicommodity flow problems."
  Operations Research, 48(2), 318-326.
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..other.operators.repair.greedy import greedy_insertion, greedy_profit_insertion
from .branching import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    BranchNode,
)
from .cutting_planes import CuttingPlaneEngine, create_cutting_plane_engine
from .master_problem import Route, VRPPMasterProblem
from .params import BPCParams
from .rcspp_dp import RCSPPSolver
from .search_strategy import create_search_strategy
from .separation import SeparationEngine
from .vrpp_model import VRPPModel

logger = logging.getLogger(__name__)


class BPCPruningException(Exception):
    """Exception raised when a node is pruned by a bound (e.g., Lagrangian)."""

    pass


# ---------------------------------------------------------------------------
# Static Helpers for Master Problem State Management
# ---------------------------------------------------------------------------


def _reset_master_constraints(master: VRPPMasterProblem) -> None:
    """Reset vehicle limits and node senses to their base problem state."""
    if master.model is None:
        return

    temp_c = master.model.getConstrByName("temp_min_vehicles")
    if temp_c:
        master.model.remove(temp_c)

    if master.vehicle_limit is not None:
        constr = master.model.getConstrByName("vehicle_limit")
        if constr:
            constr.RHS = float(master.vehicle_limit)
            constr.Sense = GRB.LESS_EQUAL

    # Reset coverage senses
    for node in range(1, master.n_nodes + 1):
        constr = master.model.getConstrByName(f"coverage_{node}")
        if constr:
            if node in master.mandatory_nodes:
                constr.Sense = GRB.EQUAL if master.strict_set_partitioning else GRB.GREATER_EQUAL
            else:
                constr.Sense = GRB.LESS_EQUAL
            constr.RHS = 1.0


def _apply_route_level_branching_filters(master: VRPPMasterProblem, bc: AnyBranchingConstraint) -> None:
    """Apply route-level feasibility filters based on branching constraints."""
    for route, var in zip(master.routes, master.lambda_vars):
        if var.UB > 0.5 and not bc.is_route_feasible(route):
            var.UB = 0.0


def _apply_branching_to_master(
    master: VRPPMasterProblem,
    branching_constraints: List[AnyBranchingConstraint],
    branching_strategy: str = "divergence",
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
    1. Incorrect LP bounds (too optimistic).
    2. Infinite branching loops (same fractional solution reappears).
    3. Invalid integer solutions.

    COLUMN BACKTRACKING & RE-ACTIVATION:
    ------------------------------------
    Since we use a global column pool, we must reset all columns to a "clean"
    state (UB=1.0) before applying the current path of branching constraints.
    Failure to do this causes columns disabled in deep branches to remain
    disabled when backtracking, leading to incorrect bounds and potential
    missed optimal solutions.

    Solution:
    ---------
    We temporarily disable violating routes by setting their upper bound to 0:
        var.UB = 0.0

    This is more efficient than:
    - Deleting and re-adding columns (expensive Gurobi operations).
    - Maintaining separate column pools per node (memory intensive).
    - Adding explicit branching constraints to Gurobi (creates dense constraint matrix).

    The routes remain in the model structure but cannot be selected in the LP solution.

    Implementation:
    ---------------
    For each route k and its corresponding Gurobi variable λ_k:
    1. Reset upper bound to 1.0 (clean state for this node).
    2. Check if route satisfies all active branching constraints (ancestors + current).
    3. If any constraint is violated, set var.UB = 0.0 to disable the route.

    Args:
        master: Master problem instance with Gurobi model.
        branching_constraints: Complete list of branching constraints from root
            to the current node (all ancestors + current node).
        branching_strategy: Current branching rule name (e.g., "ryan_foster").

    References:
    -----------
    Ryan-Foster branching (1981) is used here because it appropriately modifies the
    Resource-Constrained Shortest Path Problem (RCSPP) used for VRPP pricing by
    forbidding or enforcing pairs of nodes in the generated routes.
    """
    if master.model is None:
        return

    # Task 3/6: Harden Ryan-Foster Exactness
    if branching_strategy == "ryan_foster" and not master.strict_set_partitioning:
        raise RuntimeError(
            "Mathematical Exactness Violation: Ryan-Foster branching requires "
            "strict Set Partitioning (== 1.0). Current Master Problem allows "
            "Set Covering (>= 1.0), which can erroneously prune optimal solutions."
        )

    # MANDATORY RESET: Always clear bounds and constraint states first
    # This prevents 'constraint leaks' where branching decisions from a dead branch
    # stick around and contaminate its siblings or parent.
    for var in master.lambda_vars:
        var.UB = 1.0

    # Task 2: Reset vehicle limit and node senses to base state
    _reset_master_constraints(master)

    if not branching_constraints:
        master.model.update()
        return

    # Track effective bounds for global constraints
    max_vehicles: float = master.vehicle_limit if master.vehicle_limit is not None else float("inf")
    min_vehicles: float = 0.0
    forced_nodes: Set[int] = set()

    # Define child-local constraints
    from .branching import (
        FleetSizeBranchingConstraint,
        NodeVisitationBranchingConstraint,
    )

    # Apply all constraints in the current B&B path
    for bc in branching_constraints or []:
        # Global: Fleet Size
        if isinstance(bc, FleetSizeBranchingConstraint):
            if bc.is_upper:
                max_vehicles = min(max_vehicles, bc.limit)
            else:
                min_vehicles = max(min_vehicles, bc.limit)
        # Global: Node Visitation override
        elif isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            forced_nodes.add(bc.node)

        # Route-level filtering
        _apply_route_level_branching_filters(master, bc)

    # Apply global fleet limits to the master model
    # We can't easily add a lower bound without a new constraint if it doesn't exist.
    # We assume 'vehicle_limit' exists from build_model if specified in Params.
    vl_constr = master.model.getConstrByName("vehicle_limit")
    if vl_constr:
        # If we have a lower bound that equals the upper bound, force equality
        if min_vehicles >= max_vehicles - 1e-4:
            vl_constr.Sense = GRB.EQUAL
            vl_constr.RHS = max_vehicles
        else:
            vl_constr.Sense = GRB.LESS_EQUAL
            vl_constr.RHS = max_vehicles
            # Lower bound (>=) is rarely reached in maximization, but we add it if significant
            if min_vehicles > 0.5:
                master.model.addConstr(gp.quicksum(master.lambda_vars) >= min_vehicles, name="temp_min_vehicles")

    # Apply node visitation forcing (Sense change)
    for node in forced_nodes:
        constr = master.model.getConstrByName(f"coverage_{node}")
        if constr:
            constr.Sense = GRB.EQUAL
            constr.RHS = 1.0

    master.model.update()


def _solve_farkas_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: List[AnyBranchingConstraint],
    farkas_duals: Any,
    max_routes: int = 5,
) -> int:
    """
    Phase I Pricing: Solve subproblem with Farkas dual ray to restore feasibility.
    """
    # Task 3/6: Extract forced nodes and RF conflicts for DP enforcement
    forced_nodes: Set[int] = set()
    rf_conflicts: Dict[int, Set[int]] = {}

    from .branching import NodeVisitationBranchingConstraint, RyanFosterBranchingConstraint

    for bc in branching_constraints or []:
        if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            forced_nodes.add(bc.node)
        elif isinstance(bc, RyanFosterBranchingConstraint) and not bc.together:
            node_r, node_s = bc.node_r, bc.node_s
            rf_conflicts.setdefault(node_r, set()).add(node_s)
            rf_conflicts.setdefault(node_s, set()).add(node_r)

    # Solve subproblem
    routes: List[Route] = pricing_solver.solve(
        dual_values=farkas_duals,
        branching_constraints=branching_constraints,
        max_routes=max_routes,
        forced_nodes=forced_nodes,
        rf_conflicts=rf_conflicts,
        is_farkas=True,
    )

    added = 0
    for r in routes:
        # Task 15: Use 1e-6 to match Gurobi's default feasibility tolerance.
        # Under Farkas pricing, r.profit measures weight under the dual ray,
        # not the original route profit — any positive value helps feasibility.
        _FARKAS_TOL = 1e-6
        if r.profit > _FARKAS_TOL:
            master.add_route_as_column(r)
            added += 1
    return added


def _separate_cuts(
    master: VRPPMasterProblem,
    cut_engine: CuttingPlaneEngine,
    max_cuts: int,
    iteration: int = 0,
    node_depth: int = 0,
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
    return cut_engine.separate_and_add_cuts(master, max_cuts, iteration=iteration, node_depth=node_depth)


def _solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
    max_routes: int = 5,
    optimality_gap: float = 1e-4,
    rc_tolerance: float = 1e-5,
) -> int:
    """
    Solve the pricing subproblem and add positive reduced cost columns.

    Args:
        master: Master problem instance
        pricing_solver: RCSPP solver for pricing
        branching_constraints: Optional Ryan-Foster or NodeVisitation constraints
        max_routes: Max routes to return
        optimality_gap: Basic gap
        rc_tolerance: Minimum reduced cost to prevent epsilon deadlock

    Returns:
        Number of columns added
    """
    dual_values = master.get_reduced_cost_coefficients()

    # Task 3/6: Extract forced nodes and RF conflicts for DP enforcement
    forced_nodes: Set[int] = set()
    rf_conflicts: Dict[int, Set[int]] = {}

    from .branching import NodeVisitationBranchingConstraint, RyanFosterBranchingConstraint

    for bc in branching_constraints or []:
        if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            forced_nodes.add(bc.node)
        elif isinstance(bc, RyanFosterBranchingConstraint) and not bc.together:
            node_r, node_s = bc.node_r, bc.node_s
            rf_conflicts.setdefault(node_r, set()).add(node_s)
            rf_conflicts.setdefault(node_s, set()).add(node_r)

    # RCSPPSolver.solve() handles composite dual dictionaries and branching.
    routes: List[Route] = pricing_solver.solve(
        dual_values=dual_values,
        max_routes=max_routes,
        branching_constraints=branching_constraints,
        forced_nodes=forced_nodes,
        rf_conflicts=rf_conflicts,
    )

    if not routes:
        return 0  # No more positive reduced cost columns

    # Add new columns to master
    added = 0
    # routes is now a List[Route] from RCSPPSolver.solve
    for route in routes:
        # Task 1: Numerical Hardening (Epsilon Tail-Off Fix)
        # Only add columns if their reduced cost is strictly above rc_tolerance.
        if route.profit > rc_tolerance:
            master.add_route(route)
            added += 1
    return added


def _detect_cycles(nodes: List[int]) -> List[Tuple[int, ...]]:
    """
    Detect cyclic node sequences in a route (excluding the depot).

    Identifies segments like (i, j, k, i) where a customer node is visited
    more than once. These cycles violate the ng-route relaxation if the
    intermediate nodes are outside the neighborhood of the cycle origin.

    Args:
        nodes: Sequence of node IDs.

    Returns:
        List of node tuples forming cycles.
    """
    seen: Dict[int, int] = {}
    cycles: List[Tuple[int, ...]] = []
    for i, node in enumerate(nodes):
        if node == 0:
            continue
        if node in seen:
            # Extract the cycle from the previous occurrence to the current one
            cycle = tuple(nodes[seen[node] : i + 1])
            cycles.append(cycle)
            # Update index to detect nested or subsequent cycles involving this node
            seen[node] = i
        else:
            seen[node] = i
    return cycles


def _is_solution_integer(route_values: Dict[int, float], tol: float = 1e-6) -> bool:
    """
    Check if LP solution is integer.

    Args:
        route_values: Dictionary of route indices to their LP values
        tol: Numerical tolerance

    Returns:
        True if all values are integer (0 or 1)
    """
    for val in route_values.values():
        # Task 13: Clamp all values to [0, 1] before testing to guard against
        # slight numerical drift in Set Covering/Partitioning LP solutions.
        clamped = max(0.0, min(1.0, val))
        if abs(clamped - round(clamped)) > tol:
            return False
    return True


def _perform_strong_branching(
    candidates: List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]],
    current_node: Optional[BranchNode] = None,  # Task 11 Context
) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]], float]]:
    """
    Task 11 (SOTA): Strong Branching lookahead.
    Uses Spatial Divergence Strength as a high-fidelity proxy.
    """
    if not candidates:
        return None
    # Candidates are pre-sorted by SVRPC strength in branching.py
    return candidates[0]


def _column_generation_loop(  # noqa: C901
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    cut_engine: CuttingPlaneEngine,
    branching_constraints: Optional[List[AnyBranchingConstraint]],
    max_cg_iterations: int,
    max_cuts: int,
    time_limit: Optional[float],
    start_time: float,
    max_routes_per_pricing: int = 5,
    vehicle_limit: Optional[int] = None,
    optimality_gap: float = 1e-4,
    early_termination_gap: float = 1e-3,
    parent_basis: Optional[Any] = None,
    incumbent_value: float = -float("inf"),
    node_depth: int = 0,
) -> Tuple[float, Dict[int, float], Optional[Any]]:
    """
    Run Column Generation + Cutting Plane loop at a B&B node.
    """
    timed_out = False
    converged = False
    obj_val = -float("inf")
    route_vals: Dict[int, float] = {}
    prev_obj_val = -float("inf")
    _iteration = 0

    # Task 3: Fix Lagrangian default.
    # If no fleet limit is active, the worst-case number of vehicles is n_nodes.
    fleet_size: int = vehicle_limit if vehicle_limit is not None else master.n_nodes

    # Restore parent basis once before any LP solve at this node.
    # This warm-starts the dual simplex from the parent's optimal basis,
    # which is the primary performance benefit of Depth-First Search.
    if isinstance(parent_basis, tuple):
        master.restore_basis(*parent_basis)

    # Tasks 1 & 2: Reconcile ng-Routes with SRIs and Ryan-Foster branching.
    # We must enforce strict elementarity for nodes involved in active SRIs or
    # Ryan-Foster node-pair branching to prevent the ng-relaxation from
    # bypassing these constraints via cycles.
    elementary_nodes: Set[int] = set()

    # Extract nodes from Ryan-Foster branches
    from .branching import NodeVisitationBranchingConstraint, RyanFosterBranchingConstraint

    for bc in branching_constraints or []:
        if isinstance(bc, RyanFosterBranchingConstraint):
            elementary_nodes.add(bc.node_r)
            elementary_nodes.add(bc.node_s)
        elif isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
            # Force nodes are also strictly elementary to prevent cyclic bypass
            elementary_nodes.add(bc.node)

    # Extract nodes from active SRI subsets
    for subset in master.active_sri_cuts.keys():
        elementary_nodes.update(subset)

    if elementary_nodes:
        added_elem = pricing_solver.enforce_elementarity(list(elementary_nodes))
        if added_elem > 0:
            logger.info(
                f"[Theoretical Hardening] Enforcing strict elementarity for "
                f"{len(elementary_nodes)} nodes (SRI/Ryan-Foster parity)."
            )

    for _iteration in range(max_cg_iterations):
        if time_limit and (time.process_time() - start_time) > time_limit:
            timed_out = True
            break

        # PHASE 1: Column Generation (price until convergence)
        _inner_iter = 0
        while True:
            if time_limit and (time.process_time() - start_time) > time_limit:
                timed_out = True
                break

            # Only sift the pool after at least one LP solve has occurred at this node,
            # so the duals reflect the current node's constraint set.
            if _inner_iter > 0 and hasattr(master, "sift_global_column_pool"):
                duals = master.get_dual_values()
                if duals:
                    # Task 3: Pass composite duals to sift_global_column_pool (Stale RC Fix)
                    reactivated = master.sift_global_column_pool(
                        node_duals=duals["node_duals"],
                        rcc_duals=duals["rcc_duals"],
                        sri_duals=duals["sri_duals"],
                        edge_clique_duals=duals["edge_clique_duals"],
                        branching_constraints=branching_constraints,
                    )
                    if reactivated > 0:
                        _inner_iter += 1
                        continue

            try:
                obj_val, route_vals = master.solve_lp_relaxation()  # type: ignore[assignment]
                if obj_val is None:
                    raise RuntimeError("LP relaxation returned non-optimal status at B&B node")

                if obj_val == -float("inf"):
                    logger.info("RMP node is infeasible. Starting Phase I Farkas Pricing.")
                    added = _solve_farkas_pricing_step(
                        master,
                        pricing_solver,
                        branching_constraints,  # type: ignore[arg-type]
                        master.farkas_duals,
                    )
                    if added == 0:
                        raise RuntimeError("LP infeasible at B&B node - Farkas pricing failed to find columns")
                    _inner_iter += 1
                    continue

            except Exception as e:
                if "Farkas pricing failed" in str(e):
                    raise
                raise RuntimeError("LP relaxation failed at B&B node") from e

            added = _solve_pricing_step(
                master,
                pricing_solver,
                branching_constraints,
                max_routes=max_routes_per_pricing,
                optimality_gap=optimality_gap,
            )

            if added == 0:
                # Task 1b: Check for fractional cycles in ng-relaxation if CG has converged
                # locally. Dynamic ng-expansion serves as a lightweight cut separation.
                cycles: List[Tuple[int, ...]] = []
                for idx, val in route_vals.items():
                    if val > 0.1:
                        route = master.routes[idx]
                        route_cycles = _detect_cycles(route.nodes)
                        if route_cycles:
                            cycles.extend(route_cycles)

                if cycles:
                    added_ng = pricing_solver.expand_ng_neighborhoods(cycles)

                    # Task 1: Physically disable the cyclic columns in the Master Problem.
                    # Just expanding ng-sets is insufficient if the cyclic column remains
                    # active and dominating in the current RMP.
                    for idx, val in route_vals.items():
                        if val > 1e-4:
                            route = master.routes[idx]
                            if len(set(route.nodes)) != len(route.nodes):
                                master.lambda_vars[idx].UB = 0.0

                    if added_ng > 0:
                        logger.info(
                            f"[Dynamic ng-Expansion] Added {added_ng} neighborhood pairs "
                            f"to suppress {len(cycles)} cycles in iteration {_iteration}."
                        )
                        continue  # Re-run pricing with tightened relaxation

                # Truly converged for this cut-iteration
                break

            # Task 2b: Periodic Column Purging
            if _inner_iter > 0 and _inner_iter % 20 == 0:
                purged = master.purge_useless_columns(tolerance=-0.1)
                if purged > 0:
                    logger.info(f"Periodic purge removed {purged} useless columns.")

            _inner_iter += 1

            if hasattr(pricing_solver, "last_max_rc"):
                max_rc = getattr(pricing_solver, "last_max_rc", -float("inf"))
                effective_fleet = fleet_size if fleet_size > 0 else master.n_nodes
                lagrangian_ub = obj_val + (effective_fleet * max(0.0, max_rc))

                logger.info(
                    f"CG Iter {_iteration}.{_inner_iter}: "
                    f"RMP={obj_val:.4f}, max_rc={max_rc:.6f}, "
                    f"z_UB={lagrangian_ub:.4f}"
                )

                if lagrangian_ub < incumbent_value - 1e-6:
                    logger.info(f"Exact Pruning: z_UB {lagrangian_ub:.4f} < Incumbent {incumbent_value:.4f}")
                    raise BPCPruningException(f"Node pruned by exact Lagrangian bound: {lagrangian_ub}")

                if max_rc <= 1e-6:
                    logger.info(f"CG inner loop converged: optimality proven (max_rc {max_rc:.8f} <= 1e-6)")
                    break

        if timed_out:
            break

        # PHASE 2: Cutting Planes — Task 11: only separate when LP is optimal
        cuts_added = _separate_cuts(master, cut_engine, max_cuts, iteration=_iteration, node_depth=node_depth)

        if cuts_added == 0:
            converged = True
            break

        # Task 10 (SOTA): Tail-Off Management
        # If the objective improvement from the last iteration is negligible, stop cutting.
        obj_delta = abs(obj_val - prev_obj_val) if _iteration > 0 else float("inf")
        if obj_delta < 1e-4:
            logger.info(f"Tail-off detected: delta Z {obj_delta:.6f} < 1e-4. Stopping separation.")
            converged = True
            break
        prev_obj_val = obj_val

    # One final LP solve to get consistent obj_val / route_vals after any cuts/columns
    obj_val, route_vals = master.solve_lp_relaxation()  # type: ignore[assignment]

    # Warn only when the iteration cap truncated an unconverged loop
    if not converged and not timed_out and _iteration == max_cg_iterations - 1:
        warnings.warn(
            f"CG+Cut loop hit max_cg_iterations={max_cg_iterations} without full convergence.",
            stacklevel=3,
        )

    final_basis = master.save_basis()
    return obj_val, route_vals, final_basis


def run_bpc(  # noqa: C901
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    params: Optional[Union[BPCParams, Dict[str, Any]]] = None,
    must_go_indices: Optional[Set[int]] = None,
    vehicle_limit: Optional[int] = None,
    env: Optional[Any] = None,
    node_coords: Optional[np.ndarray] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Waste-Collecting CVRP using exact Branch-and-Price-and-Cut.

    This engine implements the Barnhart et al. (1998, OR 46(3):316-329) BPC framework with
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
        must_go_indices: Set of node indices that must be visited
        env: Gurobi environment (ignored, kept for compatibility)
        node_coords: Optional node coordinates for visualization
        recorder: Optional state recorder for tracking

    Returns:
        Tuple of (best_routes, best_cost)
    """
    start_time = time.process_time()
    n_nodes = len(dist_matrix) - 1
    m_set = must_go_indices if must_go_indices is not None else set()

    # Standardize params to BPCParams
    if params is None:
        params = BPCParams()
    elif isinstance(params, dict):
        params = BPCParams.from_config(params)

    # Configuration
    # Extract parameters from BPCParams
    max_cg_iter = params.max_cg_iterations
    max_cuts = params.max_cuts_per_iteration
    max_routes_per_pricing = params.max_routes_per_pricing
    max_bb_nodes = params.max_bb_nodes
    time_limit = params.time_limit
    search_strategy_name = params.search_strategy
    cutting_planes_name = params.cutting_planes
    branching_strategy_name = params.branching_strategy

    # 1. Initialize Master Problem
    master = VRPPMasterProblem(
        n_nodes=n_nodes,
        mandatory_nodes=m_set,
        cost_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue_per_kg=R,
        cost_per_km=C,
        vehicle_limit=vehicle_limit,
    )

    # Task 3/6: Harden Ryan-Foster Exactness
    if branching_strategy_name == "ryan_foster":
        master.strict_set_partitioning = True
        logger.info("Enforcing strict Set Partitioning (SP) for Ryan-Foster branching.")

    # 2. Initial Columns (Greedy)
    if params.profit_aware_operators:
        initial_routes_nodes = greedy_profit_insertion(
            [],
            list(range(1, n_nodes + 1)),
            dist_matrix,
            wastes,
            capacity,
            R=R,
            C=C,
            mandatory_nodes=sorted(list(m_set)),
            expand_pool=params.vrpp,
        )
    else:
        initial_routes_nodes = greedy_insertion(
            [],
            list(range(1, n_nodes + 1)),
            dist_matrix,
            wastes,
            capacity,
            mandatory_nodes=sorted(list(m_set)),
            expand_pool=params.vrpp,
        )

    initial_columns = []
    # Task 16/Fix 8: Reuse pricing_solver (remove pricing_helper instantiate)
    pricing_solver = RCSPPSolver(
        n_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        m_set,
        use_ng_routes=params.use_ng_routes,
        ng_neighborhood_size=params.ng_neighborhood_size,
    )
    for r_nodes in initial_routes_nodes:
        route_obj = pricing_solver.compute_route_details(r_nodes)
        initial_columns.append(route_obj)

    master.build_model(initial_columns)
    # Re-enabled: Column deletion is now safe due to the Global Python Column Pool
    # which preserves routes across Gurobi resets and B&B nodes.
    master.column_deletion_enabled = True

    # 2.1 Initialize 2-Phase Method
    # Start in Phase 1 (Feasibility) if there are mandatory nodes to cover.
    if len(m_set) > 0:
        master.set_phase(1)
    else:
        master.set_phase(2)

    # 3. Initialize pricing and separation
    # pricing_solver is already initialized above — reuse it.
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
    sep_engine = SeparationEngine(
        model=v_model,
        enable_heuristic_rcc_separation=params.enable_heuristic_rcc_separation,
        enable_comb_cuts=params.enable_comb_cuts,
    )

    # 4. Initialize Branch-and-Bound Tree
    bb_tree = BranchAndBoundTree(
        v_model=v_model,
        params=params,
    )

    # 5. Initialize search and cutting strategies
    search_strategy = create_search_strategy(search_strategy_name, bb_tree=bb_tree)
    cut_engine = create_cutting_plane_engine(cutting_planes_name, v_model, sep_engine)
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
        bb_tree.record_explored()  # Keep tree-level counter accurate

        # Remove node-local cuts from the previous node before solving this one.
        master.remove_local_cuts()

        # Get ALL branching constraints along the path from root to this node
        # (ancestors + this node). _apply_branching_to_master resets all UBs
        # to 1.0 and re-filters against the full constraint set, so it requires
        # the complete ancestor chain — not just this node's local constraint.
        branching_constraints = current_node.get_all_constraints()

        # 3. Apply branching constraints to master and pricing
        _apply_branching_to_master(master, branching_constraints, branching_strategy_name)

        # Run Column Generation at this node with corrected sequencing
        try:
            lp_obj, route_values, node_final_basis = _column_generation_loop(
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
                optimality_gap=params.optimality_gap,
                early_termination_gap=params.early_termination_gap,
                parent_basis=current_node.parent.lp_basis if current_node.parent else None,
                incumbent_value=bb_tree.best_integer_solution
                if bb_tree.best_integer_solution is not None
                else -float("inf"),
                node_depth=current_node.depth,
            )
            current_node.lp_basis = node_final_basis
        except BPCPruningException:
            # Node provably cannot improve the incumbent — skip without marking infeasible
            continue
        except RuntimeError as e:
            if "Conflicting must_use" in str(e):
                current_node.is_infeasible = True
                continue
            # LP infeasible at this node (regular RuntimeError)
            current_node.is_infeasible = True
            continue

        # Store LP bound
        current_node.lp_bound = lp_obj
        current_node.route_values = route_values

        # Global optimality gap check — Task 12: denominator and inclusion fix
        if bb_tree.best_integer_solution is not None:
            best_open_lp_bound = max(
                (n.lp_bound for n in bb_tree.open_nodes if n.lp_bound is not None),
                default=lp_obj,
            )
            # Always include the current node's LP value in the upper bound.
            global_upper_bound = max(best_open_lp_bound, lp_obj)
            # Use a stable denominator: at least 1.0 to avoid division by near-zero.
            denom = max(abs(bb_tree.best_integer_solution), 1.0)
            global_gap = (global_upper_bound - bb_tree.best_integer_solution) / denom

            if global_gap <= params.optimality_gap:
                logger.info(f"Global optimality gap reached: {global_gap:.6f} <= {params.optimality_gap}")
                break

        # Check if solution is integer FIRST — before pruning
        if _is_solution_integer(route_values):
            current_node.is_integer = True
            current_node.ip_solution = lp_obj

            # Update incumbent (records equal or better solutions)
            bb_tree.update_incumbent(current_node, lp_obj)
            bb_tree.prune_by_bound()

            # After recording, prune this node (no need to branch)
            continue

        # Check if we can prune by bound (strictly less than incumbent, since integer
        # solutions equal to incumbent were already handled above)
        if bb_tree.best_integer_solution is not None and lp_obj <= bb_tree.best_integer_solution:
            bb_tree.prune_by_bound()  # Ensure stale nodes are removed
            continue

        # Solution is fractional - branch
        if getattr(params, "enable_column_pool_deduplication", False):
            purged = master.deduplicate_column_pool()
            if purged > 0:
                logger.info(f"Deduplicated {purged} equivalent routes from column pool.")

        strong_candidate = None
        if getattr(params, "enable_strong_branching", False):
            candidates = bb_tree.find_strong_branching_candidates(master.routes, route_values, max_candidates=5)
            if candidates:
                strong_candidate = _perform_strong_branching(
                    candidates=candidates,
                    current_node=current_node,
                )

        children = bb_tree.branch(
            current_node,
            master.routes,
            route_values,
            mandatory_nodes=master.mandatory_nodes,
            strong_candidate=strong_candidate,
        )

        if children is None:
            # Primary branching strategy found no candidate (e.g., divergence branching
            # requires >=2 outgoing arcs at a divergence node but none exist).
            # Fall back to edge branching to avoid silently abandoning this fractional node.
            logger.warning(
                f"Primary branching strategy '{branching_strategy_name}' returned no "
                "branching candidate at a fractional node. Falling back to edge branching."
            )
            from .branching import EdgeBranching

            res = EdgeBranching.find_branching_arc(master.routes, route_values)
            if res is None:
                # Truly no branching candidate — node is degenerate, skip it.
                logger.warning("Edge branching fallback also found no candidate. Skipping node.")
                continue
            (u, v), flow = res
            left_child, right_child = EdgeBranching.create_child_nodes(current_node, u, v, flow)
            children = (left_child, right_child)

        left_child, right_child = children
        # Order child nodes on the stack so the stronger bound is explored first (LIFO)
        hint_l = left_child.lp_bound if left_child.lp_bound is not None else -float("inf")
        hint_r = right_child.lp_bound if right_child.lp_bound is not None else -float("inf")

        if hint_l < hint_r:
            # Right child is stronger (higher bound) — add left first, then right
            bb_tree.add_node(left_child)
            bb_tree.add_node(right_child)
        elif hint_r < hint_l:
            # Left child is stronger — add right first, then left
            bb_tree.add_node(right_child)
            bb_tree.add_node(left_child)
        else:
            # Task 14: Tie-break for equal LP hints.
            # Explore the child with fewer local constraints first (LP closer to parent basis).
            if len(left_child.constraints) <= len(right_child.constraints):
                bb_tree.add_node(right_child)  # pushed first = popped last
                bb_tree.add_node(left_child)  # pushed last  = popped first
            else:
                bb_tree.add_node(left_child)
                bb_tree.add_node(right_child)

    # 6. Extract best integer solution
    if bb_tree.best_integer_node is None:
        # No integer solution found - use initial greedy
        fallback_routes_profit = 0.0
        for r_nodes in initial_routes_nodes:
            route_obj = pricing_solver.compute_route_details(r_nodes)
            fallback_routes_profit += route_obj.profit
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


def _apply_reduced_cost_edge_fixing(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    z_ub: float,
    z_lb: float,
) -> int:
    """
    Task 6 (Round 3 Hardening): This feature is TEMPORARILY DISABLED.
    The current logic for reduced cost edge fixing has mathematical
    flaws regarding global vs local validity and the exact RCS formula
    used (missing dual components).

    Raising NotImplementedError to ensure it is not used in production.
    """
    raise NotImplementedError("Reduced cost edge fixing is disabled in Round 3 hardening.")
    gap = z_ub - z_lb
    if gap <= 0:
        return 0

    dual_values = master.get_reduced_cost_coefficients()
    node_duals = dual_values.get("node_duals", {})

    fixed_count = 0
    n = pricing_solver.n_nodes + 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Standard VRPP arc reduced cost under smoothed duals
            base_cost = pricing_solver.cost_matrix[i, j]
            # (Note: simpler version for now, doesn't include SRI/RCC duals on arcs)
            rc_ij = base_cost - node_duals.get(i, 0.0)

            if z_ub - max(0.0, rc_ij) < z_lb:
                # Fix edge to infinity
                pricing_solver.cost_matrix[i, j] = float("inf")
                fixed_count += 1

    if fixed_count > 0:
        logger.info(f"[SOTA Task 7] Fixed {fixed_count} edges to infinity using Lagrangian bounds (Gap: {gap:.4f}).")
    return fixed_count
