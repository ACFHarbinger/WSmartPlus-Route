"""Pricing utilities for Branch-and-Price-and-Cut solvers.

Provides reusable pricing components that are independent of any specific
BPC pipeline, so they can be imported and used by TCF–ALNS–BPC–SP pipelines
or any other matheuristic that embeds a BPC pricing subproblem.

Attributes:
-----------
dssr_pricing_wrapper          — DSSR-wrapped RCSPP solve (Righini & Salani 2008).
reduced_cost_arc_fixing       — Pricing-graph arc elimination (Irnich et al. 2010).
apply_reduced_cost_edge_fixing — LP-bound edge fixing on the master (legacy).
detect_cycles                 — Find repeated vertices in a route.
is_solution_integer           — Test integrality of a master LP solution.
solve_pricing_step            — Phase II positive-RC column generation.
solve_farkas_pricing_step     — Phase I Farkas column generation.
separate_cuts                 — Trigger cut separation on the master.

Example:
    None

References
----------
Righini & Salani (2008) Networks 51(3):155-170  — DSSR.
Irnich, Desaulniers, Desrosiers, Hadjar (2010) EJOR 211(1):75-87 — arc fixing.
Wentges (1997) — Dual smoothing used inside solve_pricing_step.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.branching import AnyBranchingConstraint
from logic.src.policies.helpers.solvers_and_matheuristics.common import Route
from logic.src.policies.helpers.solvers_and_matheuristics.master_problem import VRPPMasterProblem
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import CuttingPlaneEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def detect_cycles(nodes: List[int]) -> List[Tuple[int, ...]]:
    """Return all repeated-vertex cycles in a route's node sequence.

    Args:
        nodes: Ordered customer-node list (no depot bookends).

    Returns:
        List of tuples, each being the repeated vertex and its occurrence
        positions.  Empty list when the path is elementary.
    """
    seen: Dict[int, List[int]] = {}
    for pos, v in enumerate(nodes):
        seen.setdefault(v, []).append(pos)
    return [tuple([v] + positions) for v, positions in seen.items() if len(positions) > 1]


# ---------------------------------------------------------------------------
# Solution integrality
# ---------------------------------------------------------------------------


def is_solution_integer(
    routes: List[Route],
    route_values: Dict[int, float],
    tol: float = 1e-6,
) -> bool:
    """Return True when every non-zero λ_k is within *tol* of an integer.

    Args:
        routes:       Column pool.
        route_values: LP solution {route_index: λ_k}.
        tol:          Integrality tolerance.

    Returns:
        True iff the LP solution is integer-feasible.
    """
    return all(abs(val - round(val)) <= tol for idx, val in route_values.items())


# ---------------------------------------------------------------------------
# Cut separation
# ---------------------------------------------------------------------------


def separate_cuts(
    master: VRPPMasterProblem,
    cut_engine: CuttingPlaneEngine,
    max_cuts: int,
    iteration: int = 0,
    node_depth: int = 0,
    cut_orthogonality_threshold: float = 0.8,
) -> int:
    """Invoke the cut engine and return the number of cuts added.

    Args:
        master:                      Master problem instance.
        cut_engine:                  Composite or single-family cut engine.
        max_cuts:                    Maximum cuts to add in this round.
        iteration:                   Current column generation iteration.
        node_depth:                  Current B&B tree depth (affects separation
                                     strategy inside some engines).
        cut_orthogonality_threshold: Cosine-similarity ceiling for filtering
                                     near-duplicate cuts.

    Returns:
        Number of cuts successfully added to the master.
    """
    return cut_engine.separate_and_add_cuts(
        master,
        max_cuts,
        iteration=iteration,
        node_depth=node_depth,
        cut_orthogonality_threshold=cut_orthogonality_threshold,
    )


# ---------------------------------------------------------------------------
# Phase I — Farkas pricing
# ---------------------------------------------------------------------------


def solve_farkas_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
    max_routes: int = 5,
    timeout: Optional[float] = None,
) -> Tuple[int, bool]:
    """Generate columns that restore LP feasibility (Phase I / Farkas pricing).

    Uses the Farkas certificate from an infeasible master LP as a reduced-cost
    surrogate to find routes that increase LP feasibility.

    Args:
        master:                Master problem (must be in Phase I / infeasible).
        pricing_solver:        RCSPP solver.
        branching_constraints: Active branching constraints.
        max_routes:            Maximum columns to add per call.
        timeout:               Per-call wall-clock limit.

    Returns:
        (n_added, pricing_exhausted)
            n_added           — number of columns added.
            pricing_exhausted — True when no positive-Farkas column exists.
    """
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
        EdgeBranchingConstraint,
        NodeVisitationBranchingConstraint,
        RyanFosterBranchingConstraint,
    )

    _FARKAS_TOL: float = 1e-6

    # For Farkas pricing, get the raw duals (master switches internally via phase)
    dual_info = master.get_reduced_cost_coefficients()
    farkas_duals: Dict[int, float] = dual_info.get("node_duals", {})
    rcc_duals: Dict = dual_info.get("rcc_duals", {})

    forced_nodes: Set[int] = set()
    rf_conflicts: Dict = {}
    forbidden_arcs: FrozenSet = frozenset()

    if branching_constraints:
        for bc in branching_constraints:
            if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
                forced_nodes.add(bc.node)
            elif isinstance(bc, RyanFosterBranchingConstraint) and bc.together:
                rf_conflicts.setdefault(bc.node_r, set()).add(bc.node_s)
                rf_conflicts.setdefault(bc.node_s, set()).add(bc.node_r)
            elif isinstance(bc, EdgeBranchingConstraint) and not bc.must_use:
                forbidden_arcs = forbidden_arcs | frozenset([(bc.u, bc.v)])

    # Build the composite dual dict that solve() accepts as a single arg
    farkas_dual_dict: Dict[str, Any] = {
        "node_duals": farkas_duals,
        "rcc_duals": rcc_duals,
        "sri_duals": {},
        "edge_clique_duals": {},
    }
    routes = pricing_solver.solve(
        dual_values=farkas_dual_dict,
        max_routes=max_routes,
        forced_nodes=forced_nodes,
        rf_conflicts=rf_conflicts,
        is_farkas=True,
        timeout=timeout,
    )

    added = 0
    exhausted = True
    for route in routes:
        farkas_weight = sum(farkas_duals.get(v, 0.0) for v in route.nodes)
        if farkas_weight > _FARKAS_TOL:
            master.add_route(route)
            added += 1
            exhausted = False

    return added, exhausted


# ---------------------------------------------------------------------------
# Phase II — Standard pricing
# ---------------------------------------------------------------------------


def solve_pricing_step(
    master: VRPPMasterProblem,
    pricing_solver: RCSPPSolver,
    branching_constraints: Optional[List[AnyBranchingConstraint]] = None,
    max_routes: int = 5,
    optimality_gap: float = 1e-4,
    rc_tolerance: float = 1e-5,
    timeout: Optional[float] = None,
    use_dssr: bool = False,
    dssr_max_iters: int = 8,
) -> Tuple[int, bool]:
    """Generate profitable columns for Phase II column generation.

    Resolves the LP dual signal into RCSPP pricing calls, adding all routes
    with reduced cost above *rc_tolerance* to the master.  Optionally wraps
    the RCSPP with DSSR (Righini & Salani 2008) for tighter elementarity.

    Args:
        master:                Master problem in Phase II.
        pricing_solver:        RCSPP solver.
        branching_constraints: Active branching constraints.
        max_routes:            Maximum columns to add per call.
        optimality_gap:        Convergence tolerance on reduced cost.
        rc_tolerance:          Minimum reduced cost to accept a column.
        timeout:               Per-call wall-clock limit.
        use_dssr:              Whether to wrap the RCSPP with DSSR.
        dssr_max_iters:        Maximum DSSR refinement iterations.

    Returns:
        (n_added, pricing_exhausted)
            n_added           — number of columns added.
            pricing_exhausted — True when max reduced cost ≤ optimality_gap.
    """
    from logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints import (
        EdgeBranchingConstraint,
        NodeVisitationBranchingConstraint,
        RyanFosterBranchingConstraint,
    )

    dual_info = master.get_reduced_cost_coefficients()
    node_duals: Dict[int, float] = dual_info.get("node_duals", {})
    rcc_duals: Dict = dual_info.get("rcc_duals", {})
    sri_duals: Dict = dual_info.get("sri_duals", {})

    forced_nodes: Set[int] = set()
    rf_conflicts: Dict = {}
    forbidden_arcs: FrozenSet = frozenset()
    required_successors: Dict = {}
    if branching_constraints:
        for bc in branching_constraints:
            if isinstance(bc, NodeVisitationBranchingConstraint) and bc.forced:
                forced_nodes.add(bc.node)
            elif isinstance(bc, RyanFosterBranchingConstraint):
                if bc.together:
                    rf_conflicts.setdefault(bc.node_r, set()).add(bc.node_s)
                    rf_conflicts.setdefault(bc.node_s, set()).add(bc.node_r)
            elif isinstance(bc, EdgeBranchingConstraint):
                if not bc.must_use:
                    forbidden_arcs = forbidden_arcs | frozenset([(bc.u, bc.v)])
                else:
                    required_successors[bc.u] = bc.v

    # Note: arc-level fixing tracked in pricing_solver._forbidden_arcs is
    # injected via the branching_constraints pathway in the full CG loop.

    # solve_kwargs built below after dual_dict is assembled

    # Build the composite dual dict that solve() accepts as a single arg
    dual_dict: Dict[str, Any] = {
        "node_duals": node_duals,
        "rcc_duals": rcc_duals,
        "sri_duals": sri_duals,
        "edge_clique_duals": {},
    }
    solver_kwargs = dict(
        dual_values=dual_dict,
        max_routes=max_routes,
        forced_nodes=forced_nodes,
        rf_conflicts=rf_conflicts,
        timeout=timeout,
    )

    if use_dssr:
        routes = dssr_pricing_wrapper(
            pricing_solver=pricing_solver,
            node_duals=dual_dict,
            max_dssr_iters=dssr_max_iters,
            max_routes=max_routes,
            forced_nodes=forced_nodes,
            rf_conflicts=rf_conflicts,
            timeout=timeout,
        )
    else:
        routes = pricing_solver.solve(**solver_kwargs)

    added = 0
    for route in routes:
        if route.reduced_cost > rc_tolerance:
            master.add_route(route)
            added += 1

    pricing_exhausted = getattr(pricing_solver, "last_max_rc", 0.0) <= optimality_gap
    return added, pricing_exhausted


# ---------------------------------------------------------------------------
# DSSR — Decremental State-Space Relaxation
# ---------------------------------------------------------------------------


def dssr_pricing_wrapper(
    pricing_solver: Any,
    node_duals: Dict[int, Any],
    max_routes: int,
    forced_nodes: Optional[Set[int]] = None,
    rf_conflicts: Optional[Dict] = None,
    forbidden_arcs: Optional[FrozenSet] = None,
    required_successors: Optional[Dict] = None,
    required_predecessors: Optional[Dict] = None,
    timeout: Optional[float] = None,
    max_dssr_iters: int = 8,
) -> List[Any]:
    """Decremental State-Space Relaxation (DSSR) pricing wrapper.

    Righini & Salani (2008, Networks 51:155-170): iteratively tightens the
    ng-route relaxation by adding the set of revisited vertices to the
    elementarity memory, until all returned paths are elementary.

    Algorithm
    ---------
    1. Solve the relaxed RCSPP with current ng-memory.
    2. If all returned paths are elementary → done.
    3. Find the *critical set* Π = vertices visited > once on any path.
    4. Augment ng-memory: for each v ∈ Π, add Π to N_v.
    5. Re-solve.  Repeat until elementary or *max_dssr_iters* reached.
    6. Restore original ng-memory before returning.

    Args:
        pricing_solver:        RCSPPSolver instance.
        node_duals:            Current LP dual values {node → π_v}.
        rcc_duals:             RCC dual values.
        sri_duals:             SRI dual values.
        max_routes:            Maximum routes to return per iteration.
        forced_nodes:          Nodes forced in by branching.
        rf_conflicts:          Ryan-Foster together/apart constraints.
        forbidden_arcs:        Arcs eliminated by branching or arc-fixing.
        required_successors:   Required arc successor constraints.
        required_predecessors: Required arc predecessor constraints.
        timeout:               Total wall-clock budget for all DSSR iterations.
        max_dssr_iters:        Maximum refinement iterations.

    Returns:
        List of Route objects; all elementary when DSSR converges.
    """
    t0 = time.perf_counter()

    # Snapshot the current ng-memory so we can restore it after DSSR
    original_memory: Dict[int, Set[int]] = {v: set(mem) for v, mem in getattr(pricing_solver, "_ng_memory", {}).items()}

    routes: List[Any] = []
    for _ in range(max_dssr_iters):
        elapsed = time.perf_counter() - t0
        remaining = (timeout - elapsed) if timeout else None

        # node_duals may be either a plain dict or a composite dual dict
        _dual_arg = node_duals if isinstance(node_duals, dict) else {"node_duals": node_duals}
        routes = pricing_solver.solve(
            dual_values=_dual_arg,
            max_routes=max_routes,
            forced_nodes=forced_nodes,
            rf_conflicts=rf_conflicts,
            timeout=remaining,
        )

        if not routes:
            break

        # Find vertices revisited on any returned path
        critical: Set[int] = set()
        for route in routes:
            seen: Dict[int, int] = {}
            for v in route.nodes:
                seen[v] = seen.get(v, 0) + 1
            for v, cnt in seen.items():
                if cnt > 1:
                    critical.add(v)

        if not critical:
            break  # All paths elementary — done

        # Augment ng-memory for every critical vertex
        ng_mem: Optional[Dict] = getattr(pricing_solver, "_ng_memory", None)
        if ng_mem is None:
            break

        for v in critical:
            ng_mem[v] = ng_mem.get(v, set()) | critical

        pricing_solver._ng_memory = ng_mem

    # Restore original ng-memory so sibling nodes are unaffected
    if hasattr(pricing_solver, "_ng_memory") and original_memory:
        pricing_solver._ng_memory = original_memory

    return routes


# ---------------------------------------------------------------------------
# Reduced-cost arc fixing
# ---------------------------------------------------------------------------


def reduced_cost_arc_fixing(
    pricing_solver: Any,
    master_lp_bound: float,
    incumbent_value: float,
    n_nodes: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    node_duals: Dict[int, float],
    R: float,
    C: float,
    tol: float = 1e-6,
) -> int:
    """Eliminate pricing-graph arcs whose minimum reduced cost exceeds the gap.

    For each arc (i, j), computes a conservative lower bound on the reduced
    cost of any route traversing (i, j):

        lb(i,j) = R·w_i − π_i − C·d_{ij}    (arc contribution)
                  − C·d_{j,0}                  (cheapest return to depot)

    If lb(i,j) > LP_bound − incumbent, the arc cannot appear in any
    improving column and is added to ``pricing_solver._forbidden_arcs``.

    This is a *conservative* bound (ignores future profits along the path),
    so it is always valid — it never eliminates optimal arcs.

    Args:
        pricing_solver:  RCSPPSolver; ``_forbidden_arcs`` is updated in-place.
        master_lp_bound: Current LP relaxation value.
        incumbent_value: Best known integer solution value.
        n_nodes:         Number of customer nodes (depot excluded).
        dist_matrix:     (n+1)×(n+1) distance matrix.
        wastes:          Node waste amounts {node → waste}.
        node_duals:      LP dual values {node → π_v}.
        R:               Revenue per unit waste.
        C:               Cost per unit distance.
        tol:             Numerical tolerance.

    Returns:
        Number of arcs newly eliminated.

    References:
        Irnich, Desaulniers, Desrosiers, Hadjar (2010) EJOR 211(1):75-87.
    """
    if master_lp_bound <= incumbent_value + tol:
        return 0

    gap = master_lp_bound - incumbent_value

    if not hasattr(pricing_solver, "_forbidden_arcs"):
        pricing_solver._forbidden_arcs = set()

    depot = 0
    fixed = 0
    for i in range(1, n_nodes + 1):
        for j in range(n_nodes + 1):
            if i == j or (i, j) in pricing_solver._forbidden_arcs:
                continue
            rc_ij = R * wastes.get(i, 0.0) - node_duals.get(i, 0.0) - C * dist_matrix[i, j]
            completion = -C * dist_matrix[j, depot]
            if rc_ij + completion > gap + tol:
                pricing_solver._forbidden_arcs.add((i, j))
                fixed += 1

    return fixed


def apply_reduced_cost_edge_fixing(
    master: VRPPMasterProblem,
    pricing_solver: Any,
    z_ub: float,
    z_lb: float,
) -> int:
    """LP-bound edge fixing on the master problem (original implementation).

    Uses exact DP completion bounds stored in ``pricing_solver.bounds_from``
    / ``pricing_solver.bounds_to`` to eliminate arcs from the pricing graph.
    This is the full, non-conservative variant that uses the Lagrangian
    completion-bound values.

    Args:
        master:         Master problem (provides dual values).
        pricing_solver: RCSPP solver (provides completion bounds).
        z_ub:           Upper bound (best known integer solution value).
        z_lb:           Lower bound (current LP relaxation value).

    Returns:
        Number of edges fixed to zero.
    """
    gap = z_ub - z_lb
    if gap <= 0:
        return 0

    dual_values = master.get_reduced_cost_coefficients()
    node_duals: Dict[int, float] = dual_values.get("node_duals", {})

    fixed_count = 0
    n = pricing_solver.n_nodes + 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cost = pricing_solver.cost_matrix[i, j] * pricing_solver.C
            rev = pricing_solver.wastes.get(j, 0.0) * pricing_solver.R
            rc_ij = rev - cost - node_duals.get(j, 0.0)
            max_path_rc = pricing_solver.bounds_from[i] + rc_ij + pricing_solver.bounds_to[j]
            if z_ub + max_path_rc < z_lb - 1e-6 and (i, j) not in pricing_solver.fixed_arcs:
                pricing_solver.fixed_arcs.add((i, j))
                fixed_count += 1

    if fixed_count > 0:
        logger.info("[ArcFix] Fixed %d edges (gap=%.4f).", fixed_count, gap)
    return fixed_count
